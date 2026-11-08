package landing

import (
	"io"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestALandingStaysInsideItsStepsFiles holds a step that lists its files to them: on Nov 7 a helper's side effect
// rewrote the trunk's hook setting, outside any step's work. A change to a file the step does not list is refused,
// the refusal printing the path; a change inside the list, its tests included, with the plan beside it, is taken; and
// a step that lists nothing takes any change.
func TestALandingStaysInsideItsStepsFiles(t *testing.T) {
	t.Parallel()
	taken := func(fixture *sliceFixture) error {
		t.Helper()
		l := &landing{Options: Options{Repo: fixture.repo, Ref: "row/do", Log: io.Discard}, scratch: t.TempDir(), details: map[string]string{}}
		testcheck.NoError(t, l.parents(t.Context()))
		return l.shipSet(t.Context())
	}
	fixture := newRowFixture(t, map[string]string{"cmd/x/x.go": "package main\n", "internal/y/y.go": "package y\n"})
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"cmd/x"} })
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "a step that lists its files")

	fixture.write("cmd/x/x.go", "package main\n\nfunc main() {}\n")
	fixture.write("internal/y/y.go", "package y\n\nfunc y() {}\n")
	if err := taken(fixture); err == nil || !strings.Contains(err.Error(), "internal/y/y.go") || strings.Contains(err.Error(), "cmd/x/x.go,") {
		t.Fatalf("a change outside the step's files ended as %v, want it refused printing internal/y/y.go alone", err)
	}
	fixture.git("checkout", "-q", "HEAD", "--", "internal/y/y.go")
	fixture.write("cmd/x/x_test.go", "package main\n")
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Rationale = "noted" })
	testcheck.NoError(t, taken(fixture))

	open := newRowFixture(t, map[string]string{"internal/y/y.go": "package y\n"})
	open.write("internal/y/y.go", "package y\n\nfunc y() {}\n")
	testcheck.NoError(t, taken(open))
}
