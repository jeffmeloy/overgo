package landing

import (
	"bytes"
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
