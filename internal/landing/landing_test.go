package landing

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestLandingFixRegeneratesTheAPIManifest holds the fix stage to the
// derived API manifest: a Go change regenerates it with the repository's own
// command, and the regenerated file joins the ship set the checks validate.
func TestLandingFixRegeneratesTheAPIManifest(t *testing.T) {
	t.Parallel()
	fixture := newSliceFixture(t, map[string]string{
		"cmd/api-manifest/main.go": "package main\n\nimport \"os\"\n\nfunc main() { os.WriteFile(\"docs/api_manifest.json\", []byte(\"regenerated\\n\"), 0o644) }\n",
		"internal/p/p.go":          "package p\n",
	})
	fixture.write("internal/p/p.go", "// Package p is the fixture.\npackage p\n")
	l := &landing{Options: Options{Repo: fixture.repo, Log: io.Discard}, details: map[string]string{}}
	if err := l.shipSet(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := l.fix(t.Context()); err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(filepath.Join(fixture.repo, "docs", "api_manifest.json")); err != nil || string(data) != "regenerated\n" {
		t.Fatalf("the manifest after the fix stage = %q, %v", data, err)
	}
	if !slices.Contains(l.paths, "docs/api_manifest.json") {
		t.Fatalf("the ship set after the fix stage = %q, want the regenerated manifest in it", l.paths)
	}
}

// TestLandingCommitsTheBytesItValidated holds the commit to the ship set
// the checks validated: bytes edited after validation are refused with HEAD
// where it was, a commit that fails gives the row back in the plan, and the
// validated bytes land once nothing moved.
func TestLandingCommitsTheBytesItValidated(t *testing.T) {
	t.Parallel()
	fixture := newSliceFixture(t, map[string]string{"a.txt": "base\n"})
	fixture.git("config", "user.name", "fixture")
	fixture.git("config", "user.email", "fixture@example.com")
	planPath := filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))
	document := plan.Plan{Campaign: "fixture", Doctrine: "fixture", Items: []plan.Item{{ID: "row", Title: "row", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Title: "row", Status: plan.StatusOpen, Verify: "true"}}}}}
	if err := plan.Save(planPath, document); err != nil {
		t.Fatal(err)
	}
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "plan")
	head := fixture.git("rev-parse", "HEAD")
	validated := func() *landing {
		t.Helper()
		l := &landing{Options: Options{Repo: fixture.repo, Ref: "row/do", Message: []byte("change"), Log: io.Discard},
			item: "row", step: "do", head: head, details: map[string]string{}}
		if err := l.shipSet(t.Context()); err != nil {
			t.Fatal(err)
		}
		return l
	}
	original, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatal(err)
	}

	fixture.write("a.txt", "validated\n")
	l := validated()
	fixture.write("a.txt", "edited after validation\n")
	if _, err := l.commit(t.Context()); err == nil || !strings.Contains(err.Error(), "changed during the landing") {
		t.Fatalf("bytes edited after validation landed: %v", err)
	}
	if now := fixture.git("rev-parse", "HEAD"); now != head {
		t.Fatalf("a refused commit moved HEAD to %s", now)
	}

	fixture.write("a.txt", "validated\n")
	l = validated()
	hook := filepath.Join(fixture.repo, ".git", "hooks", "pre-commit")
	fixture.write(".git/hooks/pre-commit", "#!/bin/sh\nexit 1\n")
	if _, err := l.commit(t.Context()); err == nil {
		t.Fatal("a commit the hook refused reported success")
	}
	if restored, err := os.ReadFile(planPath); err != nil || !bytes.Equal(restored, original) {
		t.Fatalf("a failed commit left the plan without its row: %v", err)
	}

	if err := os.Remove(hook); err != nil {
		t.Fatal(err)
	}
	l = validated()
	commit, err := l.commit(t.Context())
	if err != nil {
		t.Fatalf("the validated bytes did not land: %v", err)
	}
	if landed := fixture.git("show", commit+":a.txt"); landed != "validated" {
		t.Fatalf("landed a.txt = %q, want the validated bytes", landed)
	}
}
