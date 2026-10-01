package landing

import (
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// newRowFixture commits a module whose plan holds one open row, row/do,
// whose check passes.
func newRowFixture(t *testing.T, stub map[string]string) *sliceFixture {
	t.Helper()
	fixture := newSliceFixture(t, stub)
	fixture.git("config", "user.name", "fixture")
	fixture.git("config", "user.email", "fixture@example.com")
	document := plan.Plan{Campaign: "fixture", Doctrine: "fixture", Items: []plan.Item{{ID: "row", Title: "row", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Title: "row", Status: plan.StatusOpen, Verify: "true"}}}}}
	if err := plan.Save(fixture.planPath(), document); err != nil {
		t.Fatal(err)
	}
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "plan")
	return fixture
}

func (f *sliceFixture) planPath() string {
	return filepath.Join(f.repo, filepath.FromSlash(plan.Path))
}

// candidate takes the dirty tree as a landing of row/do at HEAD takes it.
func (f *sliceFixture) candidate() *landing {
	f.t.Helper()
	l := &landing{Options: Options{Repo: f.repo, Ref: "row/do", Message: []byte("change"), Log: io.Discard},
		item: "row", step: "do", scratch: f.t.TempDir(), details: map[string]string{}}
	if err := l.parents(f.t.Context()); err != nil {
		f.t.Fatal(err)
	}
	if err := l.shipSet(f.t.Context()); err != nil {
		f.t.Fatal(err)
	}
	return l
}

// hook installs the reference-transaction hook, which git runs as a ref
// moves: script runs where the publication's compare-and-set does, as a
// worker writing into the tree at that moment would.
func (f *sliceFixture) hook(script string) {
	f.t.Helper()
	path := filepath.Join(f.repo, ".git", "hooks", "reference-transaction")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n[ \"$1\" = prepared ] || exit 0\n"+script), 0o755); err != nil {
		f.t.Fatal(err)
	}
	f.t.Cleanup(func() { os.Remove(path) })
}

func (f *sliceFixture) read(name string) string {
	f.t.Helper()
	data, err := os.ReadFile(filepath.Join(f.repo, filepath.FromSlash(name)))
	if err != nil {
		f.t.Fatal(err)
	}
	return string(data)
}

// TestLandingFixRegeneratesTheAPIManifest holds the fix stage to the
// derived API manifest: a Go change regenerates it with the repository's own
// command, and the regenerated file joins the ship set the checks validate.
func TestLandingFixRegeneratesTheAPIManifest(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{
		"cmd/api-manifest/main.go": "package main\n\nimport \"os\"\n\nfunc main() { os.WriteFile(\"docs/api_manifest.json\", []byte(\"regenerated\\n\"), 0o644) }\n",
		"internal/p/p.go":          "package p\n",
	})
	fixture.write("internal/p/p.go", "// Package p is the fixture.\npackage p\n")
	l := fixture.candidate()
	if err := l.fix(t.Context()); err != nil {
		t.Fatal(err)
	}
	if data := fixture.read("docs/api_manifest.json"); data != "regenerated\n" {
		t.Fatalf("the manifest after the fix stage = %q", data)
	}
	if !slices.Contains(l.paths, "docs/api_manifest.json") {
		t.Fatalf("the ship set after the fix stage = %q, want the regenerated manifest in it", l.paths)
	}
	if manifest := fixture.git("cat-file", "blob", l.candidate+":docs/api_manifest.json"); manifest != "regenerated" {
		t.Fatalf("the candidate after the fix stage holds manifest %q, want the regenerated one", manifest)
	}
}

// TestLandingPublishesTheCheckedCandidate holds the commit to the candidate
// the checks validated: files changed after the candidate is taken are
// refused and named with HEAD where it was, an edit made while the commit
// publishes cannot enter it, and the commit holds the candidate's bytes and
// its plan advanced past the row, onto the HEAD the landing found.
func TestLandingPublishesTheCheckedCandidate(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n"})
	original := fixture.read(plan.Path)
	fixture.write("a.txt", "checked\n")
	l := fixture.candidate()
	if l.verifier != "true" {
		t.Fatalf("the row's check = %q, want the candidate plan's", l.verifier)
	}

	fixture.write("a.txt", "edited after the candidate\n")
	fixture.write("b.txt", "added after the candidate\n")
	fixture.write(plan.Path, strings.Replace(original, `"verify": "true"`, `"verify": "false"`, 1))
	_, err := l.commit(t.Context())
	if err == nil || !strings.Contains(err.Error(), "a.txt") || !strings.Contains(err.Error(), "b.txt") || !strings.Contains(err.Error(), plan.Path) {
		t.Fatalf("a tree changed after the candidate was taken landed, or the refusal does not name its paths: %v", err)
	}
	if now := fixture.git("rev-parse", "HEAD"); now != l.head {
		t.Fatalf("a refused landing moved HEAD to %s", now)
	}

	fixture.write(plan.Path, original)
	fixture.write("a.txt", "checked\n")
	if err := os.Remove(filepath.Join(fixture.repo, "b.txt")); err != nil {
		t.Fatal(err)
	}
	fixture.hook("echo late > a.txt\n")
	commit, err := l.commit(t.Context())
	if err != nil {
		t.Fatalf("the checked candidate did not land: %v", err)
	}
	if landed := fixture.git("show", commit+":a.txt"); landed != "checked" {
		t.Fatalf("landed a.txt = %q, want the checked bytes", landed)
	}
	if parent := fixture.git("rev-parse", commit+"^"); parent != l.head {
		t.Fatalf("landed onto %s, want the HEAD the landing found, %s", parent, l.head)
	}
	if published, err := plan.Parse([]byte(fixture.git("cat-file", "blob", commit+":"+plan.Path))); err != nil || len(published.Items) != 0 {
		t.Fatalf("the landed plan = %+v, %v; want the row advanced", published.Items, err)
	}
	if status := fixture.git("status", "--porcelain"); status != "M a.txt" {
		t.Fatalf("status after the landing = %q, want only the late edit, uncommitted", status)
	}
}

// TestLandingLeavesAConcurrentPlanEditAlone holds a landing to writing the
// plan only where it holds what the landing read: a publication that fails
// leaves a plan edited meanwhile as edited, a HEAD that moved refuses the
// compare-and-set with the row still in the plan, and a commit that lands
// while the plan is edited leaves the edit and says so.
func TestLandingLeavesAConcurrentPlanEditAlone(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n"})
	original := fixture.read(plan.Path)
	fixture.write("a.txt", "checked\n")
	l := fixture.candidate()

	fixture.hook("echo edited > " + plan.Path + "\nexit 1\n")
	if _, err := l.commit(t.Context()); err == nil {
		t.Fatal("a refused publication reported success")
	}
	if held := fixture.read(plan.Path); held != "edited\n" {
		t.Fatalf("a failed publication overwrote the plan edited meanwhile with %q", held)
	}
	if now := fixture.git("rev-parse", "HEAD"); now != l.head {
		t.Fatalf("a refused publication moved HEAD to %s", now)
	}

	fixture.write(plan.Path, original)
	if err := os.Remove(filepath.Join(fixture.repo, ".git", "hooks", "reference-transaction")); err != nil {
		t.Fatal(err)
	}
	fixture.git("commit", "-q", "--allow-empty", "-m", "moved")
	moved := fixture.git("rev-parse", "HEAD")
	if _, err := l.commit(t.Context()); err == nil {
		t.Fatal("a publication onto a HEAD that moved reported success")
	}
	if now := fixture.git("rev-parse", "HEAD"); now != moved {
		t.Fatalf("a refused compare-and-set moved HEAD from %s to %s", moved, now)
	}
	if held := fixture.read(plan.Path); held != original {
		t.Fatalf("a refused publication took the row from the plan: %q", held)
	}

	l = fixture.candidate()
	fixture.hook("echo edited > " + plan.Path + "\n")
	commit, err := l.commit(t.Context())
	if err == nil || commit == "" || !strings.Contains(err.Error(), "left as edited") {
		t.Fatalf("a landing over a plan edited meanwhile = %q, %v; want the commit and the conflict", commit, err)
	}
	if held := fixture.read(plan.Path); held != "edited\n" {
		t.Fatalf("a landed commit overwrote the plan edited meanwhile with %q", held)
	}
}

// TestLandingPublishesAMergeAsTheMerge holds a landing over a merge in
// progress to the merge: its head joins HEAD as a parent and the merge ends.
func TestLandingPublishesAMergeAsTheMerge(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n"})
	side := fixture.git("commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "side")
	fixture.git("merge", "-q", "--no-ff", "--no-commit", side)
	fixture.write("a.txt", "merged\n")
	l := fixture.candidate()
	commit, err := l.commit(t.Context())
	if err != nil {
		t.Fatalf("the merge did not land: %v", err)
	}
	if parents := fixture.git("rev-list", "--parents", "-n", "1", commit); parents != commit+" "+l.head+" "+side {
		t.Fatalf("the landed merge's parents = %q, want HEAD then %s", parents, side)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, ".git", "MERGE_HEAD")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("the landed merge left the merge in progress: %v", err)
	}
}
