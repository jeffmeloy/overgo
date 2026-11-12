package landing

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestAChecksFormIsReadBeforeItRuns holds a landing to reading each step's check before any long stage, card or not
// (cheap-first/form): a cheap stage named form comes before every package's tests and the lanes; a check that needs
// model files alone runs on the host, before the lanes, while one that sets OVERGO_CUDA_TEST=1 keeps its run after
// them; the form stage refuses a check naming a package that holds no test of its name; and a change whose tests do
// not build without it, on a step without its note, is refused there, not after the long stages.
func TestAChecksFormIsReadBeforeItRuns(t *testing.T) {
	t.Parallel()
	stage := func(checks []check, name string) int {
		return slices.IndexFunc(checks, func(check check) bool { return check.name == name })
	}
	host := "go test ./internal/devicebudget -count=1 -run '^TestADeclaredDeviceBudgetBoundsWhatACardWould$'"
	models := "OVERGO_DATA_ROOT=C:/data go test -tags modeltest ./cmd/validate -count=1 -run '^TestThreeWorkloadsAreReadUnderSmallerCards$' -timeout 120m"
	card := "OVERGO_CUDA_TEST=1 OVERGO_DATA_ROOT=C:/data go test -tags modeltest ./cmd/loop -count=1 -run '^TestTheLocalWorkerLandsStepsUnaided$' -timeout 120m"
	for _, verify := range []string{host, models, card} {
		order := (&landing{verifiers: []string{verify}}).checks()
		var names []string
		for _, check := range order {
			names = append(names, check.name)
		}
		if form := stage(order, "form"); form < 0 || !order[form].cheap || form > stage(order, "test") || form > stage(order, "lanes") {
			t.Errorf("a landing of %q orders its stages %v; want a cheap form stage before the tests and the lanes", verify, names)
		}
		if late := stage(order, "verify") > stage(order, "lanes"); late != (verify == card) {
			t.Errorf("a landing of %q orders its stages %v; want its run after the lanes only where it sets OVERGO_CUDA_TEST=1", verify, names)
		}
	}

	form := func(t *testing.T, verify, source, test string) error {
		t.Helper()
		fixture := newRowFixture(t, map[string]string{"p/p.go": undoneValue, "q/q.go": "// Package q is a fixture.\npackage q\n"})
		editPlan(t, fixture, func(document *plan.Plan) {
			document.Items[0].Budget.Nodes = 100
			document.Items[0].Steps[0].Verify = verify
		})
		fixture.git("add", "-A")
		fixture.git("commit", "-q", "-m", "the row's check")
		fixture.write("p/p.go", source)
		fixture.write("p/p_test.go", "package p\n\nimport \"testing\"\n\n"+test)
		l := fixture.candidate()
		at := stage(l.checks(), "form")
		if at < 0 {
			t.Fatal("no stage of the landing is named form")
		}
		return l.checks()[at].run(t.Context())
	}
	t.Run("a check whose test is there and builds without the change", func(t *testing.T) {
		t.Parallel()
		if err := form(t, undoneCheck, strings.Replace(undoneValue, "return 1", "return 2", 1), caughtTest); err != nil {
			t.Fatalf("the form stage refused a check whose test is in its package and builds without the change: %v", err)
		}
	})
	t.Run("a package without the check's test", func(t *testing.T) {
		t.Parallel()
		err := form(t, "go test ./p ./q -run '^TestValue$' -count=1", strings.Replace(undoneValue, "return 1", "return 2", 1), caughtTest)
		if err == nil || !strings.Contains(err.Error(), "./q holds no test") {
			t.Fatalf("the form stage gave %v for a check naming ./q, which holds no TestValue; want it refused naming ./q", err)
		}
	})
	t.Run("tests that do not build without the change", func(t *testing.T) {
		t.Parallel()
		err := form(t, undoneCheck, undoneValue+twice, unbuiltTest)
		if err == nil || !strings.Contains(err.Error(), "No boundary test:") {
			t.Fatalf("the form stage gave %v for a change whose test does not build without it, on a step with no note; want it refused asking for the note", err)
		}
	})
}
