package landing

import (
	"io"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestAnAcceptedCheckRunsAsAccepted holds a landing to the check its step was accepted with (SQA-378): a check naming
// ./internal/a and ./internal/b, of which neither holds its test, is not passed by a test of the same name that only
// ./internal/c declares, though that test fails without the change and passes with it. A landing that ran the test
// wherever it was declared granted a pass to a different check than the accepted one; a legitimate move of a test is
// a check changed in the plan, recorded as one.
func TestAnAcceptedCheckRunsAsAccepted(t *testing.T) {
	t.Parallel()
	const (
		before   = "package c\n\n// C is what the change sets.\nconst C = 0\n"
		after    = "package c\n\n// C is what the change sets.\nconst C = 1\n"
		test     = "package c\n\nimport \"testing\"\n\nfunc TestX(t *testing.T) {\n\tif C != 1 {\n\t\tt.Fatal(C)\n\t}\n}\n"
		accepted = "go test ./internal/a ./internal/b -count=1 -run '^TestX$'"
	)
	fixture := newRowFixture(t, map[string]string{
		"internal/a/a.go": "package a\n", "internal/b/b.go": "package b\n",
		"internal/c/c.go": before, "internal/c/c_test.go": test,
	})
	fixture.commitPlan(rowPlan(0, [2]string{"do", accepted}))
	testcheck.NoError(t, plan.Save(fixture.planPath(), rowPlan(0, [2]string{"do", accepted})))
	fixture.write("internal/c/c.go", after)
	l := fixture.candidateOf("row/do")
	l.Log = io.Discard
	err := l.verify(t.Context())
	if err == nil {
		t.Fatal("a check naming ./internal/a and ./internal/b passed on a test only ./internal/c declares")
	}
	if !strings.Contains(err.Error(), "matched no passing test") {
		t.Errorf("a check whose packages hold no test of its name was refused for %v; want it refused as matching no passing test", err)
	}
}
