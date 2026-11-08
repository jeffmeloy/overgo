package landing

import (
	"bytes"
	"io"
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestALandingCommitsOnlyItsStepsFiles holds a landing to the files its steps list: on 2026-11-08 another session's
// untracked document in the lead's checkout stopped every landing for forty minutes, since a landing took the whole
// folder though each step names what it may change. A landing of a step that lists its files commits those, its tests
// among them, with the plan; a changed file and an untracked one the step does not list stay where they are, outside
// the commit, and the landing names them; and a step that lists nothing takes the whole tree as before.
func TestALandingCommitsOnlyItsStepsFiles(t *testing.T) {
	t.Parallel()
	taken := func(fixture *sliceFixture) (*landing, string) {
		t.Helper()
		var output bytes.Buffer
		l := &landing{Options: Options{Repo: fixture.repo, Ref: "row/do", Log: &output}, scratch: t.TempDir(), details: map[string]string{}}
		testcheck.NoError(t, l.parents(t.Context()))
		testcheck.NoError(t, l.shipSet(t.Context()))
		return l, output.String()
	}
	committed := func(fixture *sliceFixture, l *landing) []string {
		return strings.Fields(fixture.git("ls-tree", "-r", "--name-only", l.candidate))
	}
	fixture := newRowFixture(t, map[string]string{"cmd/x/x.go": "package main\n", "internal/y/y.go": "package y\n"})
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"cmd/x"} })
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "a step that lists its files")

	fixture.write("cmd/x/x.go", "package main\n\nfunc main() {}\n")
	fixture.write("cmd/x/x_test.go", "package main\n")
	fixture.write("internal/y/y.go", "package y\n\nfunc y() {}\n")
	fixture.write("docs/another_sessions_plan.md", "# not this step's\n")
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Rationale = "noted" })
	l, output := taken(fixture)
	files := committed(fixture, l)
	if !slices.Contains(files, "cmd/x/x_test.go") || slices.Contains(files, "docs/another_sessions_plan.md") ||
		fixture.git("show", l.candidate+":internal/y/y.go") != "package y" || fixture.git("show", l.candidate+":cmd/x/x.go") != "package main\n\nfunc main() {}" {
		t.Fatalf("the commit holds %v, want the step's files changed and nothing else", files)
	}
	if !strings.Contains(output, "outside the commit") || !strings.Contains(output, "internal/y/y.go") || !strings.Contains(output, "docs/another_sessions_plan.md") {
		t.Fatalf("the landing said\n%s\nwant the paths it left out named", output)
	}
	if fixture.read("docs/another_sessions_plan.md") != "# not this step's\n" || fixture.read("internal/y/y.go") != "package y\n\nfunc y() {}\n" {
		t.Fatal("a path outside the commit was changed")
	}

	open := newRowFixture(t, map[string]string{"internal/y/y.go": "package y\n"})
	open.write("internal/y/y.go", "package y\n\nfunc y() {}\n")
	if l, _ := taken(open); open.git("show", l.candidate+":internal/y/y.go") != "package y\n\nfunc y() {}" {
		t.Fatal("a step that lists nothing did not take the whole tree")
	}
}

// TestALandingTakesWhatItsGeneratorsDerive holds the scope to the change's own work: on 2026-11-08
// media-user-size/preset-record added exported names, the landing regenerated docs/api_manifest.json from them, and
// the scope refused the manifest the landing had written itself. A file a generator of the repository derives is taken
// beside the step's files; the same file in a repository with no such generator is no part of the step, and stays
// outside the commit.
func TestALandingTakesWhatItsGeneratorsDerive(t *testing.T) {
	t.Parallel()
	for _, generated := range []bool{true, false} {
		files := map[string]string{"cmd/x/x.go": "package main\n", "docs/api_manifest.json": "{}\n"}
		if generated {
			files["cmd/api-manifest/main.go"] = "package main\n"
		}
		fixture := newRowFixture(t, files)
		editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"cmd/x"} })
		fixture.git("add", "-A")
		fixture.git("commit", "-q", "-m", "a step that lists its files")
		fixture.write("cmd/x/x.go", "package main\n\nfunc main() {}\n")
		fixture.write("docs/api_manifest.json", "{\"binaries\": []}\n")
		l := &landing{Options: Options{Repo: fixture.repo, Ref: "row/do", Log: io.Discard}, scratch: t.TempDir(), details: map[string]string{}}
		testcheck.NoError(t, l.parents(t.Context()))
		testcheck.NoError(t, l.shipSet(t.Context()))
		if taken := fixture.git("show", l.candidate+":docs/api_manifest.json") != "{}"; taken != generated {
			t.Errorf("with the manifest's generator %t, the manifest was taken %t, want it taken only where the generator is", generated, taken)
		}
	}
}

// TestAFailedLandingNamesThePathsItLeftOut holds a landing that left out what its change needs to saying so: on
// 2026-11-08 seven steps were retired and filed again because their lists left out a directory the change needed, and
// since a landing commits only its steps' files such a change fails a check that names no path. A change to cmd/x calls
// a function it adds to internal/y, which the step does not list: the refusal names internal/y/y.go and the plan
// -setfiles command that lists internal/y, and once the plan lists it the change lands with it.
func TestAFailedLandingNamesThePathsItLeftOut(t *testing.T) {
	t.Parallel()
	// The module is overgo's, whose import paths the rule that an export has a consumer reads.
	fixture := newRowFixture(t, map[string]string{"go.mod": "module overgo\n\ngo 1.26\n", "cmd/x/x.go": "package main\n\nfunc main() {}\n", "internal/y/y.go": "package y\n"})
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"cmd/x"} })
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "a step that lists its files")
	fixture.write("internal/y/y.go", "package y\n\n// Name is what x prints.\nfunc Name() string { return \"y\" }\n")
	fixture.write("cmd/x/x.go", "package main\n\nimport \"overgo/internal/y\"\n\nfunc main() { println(y.Name()) }\n")
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")
	_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Message: message, Log: io.Discard, Preflight: true})
	command := `go run ./cmd/plan -setfiles -files cmd/x,internal/y -reason "the change needs internal/y" row do`
	if err == nil || !strings.Contains(err.Error(), "left out internal/y/y.go") || !strings.Contains(err.Error(), command) {
		t.Fatalf("the landing ended as %v, want it refused naming internal/y/y.go and %s", err, command)
	}
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"cmd/x", "internal/y"} })
	landed := testcheck.Must(Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Message: message, Log: io.Discard}))(t)
	if !strings.Contains(fixture.git("show", landed+":internal/y/y.go"), "func Name") {
		t.Fatal("once the plan listed internal/y, the landing did not take it")
	}
}
