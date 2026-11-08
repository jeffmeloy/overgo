package landing

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestTheChecksReadTheCommitNotTheFolder holds the gate to the commit it makes: on 2026-11-08 another session's
// document in the lead's checkout failed the reader check and the stray-documents rule of every landing, since the
// stages read the folder. Beside a change to cmd/x, the folder holds another session's Go file that does not compile
// and its ignored draft. The change lands, its vet and tests reading the commit in a worktree of its own, and the draft
// is named, not refused; a preflight of the next change reads in the same worktree, since go's caches key what they
// keep by a package's directory.
func TestTheChecksReadTheCommitNotTheFolder(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"cmd/x/x.go": "package main\n\nfunc main() {}\n", ".gitignore": "docs/draft.md\n"})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Steps[0].Files = []string{"cmd/x"}
		document.Items[0].Steps = append(document.Items[0].Steps, plan.Step{ID: "next", Title: "next", Status: plan.StatusOpen, Verify: "true", Files: []string{"cmd/x"}})
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "steps that list their files")
	fixture.write("cmd/x/x.go", "package main\n\nfunc main() { println() }\n")
	fixture.write("cmd/x/x_test.go", "package main\n\nimport \"testing\"\n\nfunc TestX(t *testing.T) {}\n")
	fixture.write("internal/z/z.go", "package z\n\nfunc broken( {\n")
	fixture.write("docs/draft.md", "# another session's plan\n")
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")

	var said bytes.Buffer
	landed := testcheck.Must(Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Message: message, Log: &said}))(t)
	if !strings.Contains(said.String(), "docs/draft.md") || fixture.git("show", landed+":cmd/x/x_test.go") == "" {
		t.Fatalf("the landing said\n%s\nwant the change landed and the draft named", said.String())
	}
	fixture.write("cmd/x/x.go", "package main\n\nfunc main() { println(1) }\n")
	testcheck.Must(Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/next", Message: message, Log: &said, Preflight: true}))(t)
	if trees := fixture.git("worktree", "list", "--porcelain"); strings.Count(trees, "worktree ") != 2 ||
		!strings.Contains(trees, processlock.StateDirectory+"/"+candidateCheckout+"0") {
		t.Fatalf("the worktrees are\n%s\nwant the checkout and the one both runs read, kept in its process state", trees)
	}
}

// TestTheGeneratorsDeriveFromTheCommit holds what the fix stage regenerates to the commit's source: on 2026-11-08 the
// API manifest regenerated from the folder carried the identity of a file the commit left out, and the commit's own
// test refused it. The repository's manifest command derives the manifest from a file outside the step's files that the
// folder holds changed; the manifest the commit takes, and the checkout keeps, is derived from that file as the commit
// holds it, and the worktree it was derived in resolves the machine's data roots as the checkout does.
func TestTheGeneratorsDeriveFromTheCommit(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{
		"cmd/api-manifest/main.go": "package main\n\nimport \"os\"\n\nfunc main() {\n\tsource, _ := os.ReadFile(\"internal/y/y.go\")\n\tos.WriteFile(\"docs/api_manifest.json\", source, 0o644)\n}\n",
		"cmd/x/x.go":               "package main\n\nfunc main() {}\n",
		"internal/y/y.go":          "package y\n",
		"docs/api_manifest.json":   "package y\n",
		".gitignore":               "local-models.json\n",
	})
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"cmd/x"} })
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "a step that lists its files")
	fixture.write("local-models.json", "{\"models\": \"elsewhere\"}\n")
	fixture.write("cmd/x/x.go", "package main\n\nfunc main() { println() }\n")
	fixture.write("internal/y/y.go", "package y\n\nfunc y() {}\n")
	l := fixture.candidate()
	testcheck.NoError(t, l.fix(t.Context()))
	if derived := fixture.git("show", l.candidate+":docs/api_manifest.json"); derived != "package y" || fixture.read("docs/api_manifest.json") != "package y\n" {
		t.Fatalf("the commit takes the manifest %q and the checkout keeps %q, want both derived from the commit's internal/y/y.go", derived, fixture.read("docs/api_manifest.json"))
	}
	if roots, err := os.ReadFile(filepath.Join(l.tree, "local-models.json")); err != nil || string(roots) != fixture.read("local-models.json") {
		t.Fatalf("the worktree's data roots are %q, %v, want the checkout's", roots, err)
	}
}
