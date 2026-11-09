package landing

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

const (
	// undoneValue is the fixture's production Go before the change, and undoneCheck the row's check.
	undoneValue = "// Package p is a fixture.\npackage p\n\n// Value is the fixture's value.\nfunc Value() int { return 1 }\n"
	undoneCheck = "go test ./p -run '^TestValue$' -count=1"
	// caughtTest asserts the changed value, which the value before the change fails; unbuiltTest calls a function the
	// change adds, so it does not build without the change.
	caughtTest  = "func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 2 {\n\t\tt.Fatal(Value())\n\t}\n}\n"
	unbuiltTest = "func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Twice() != 2 {\n\t\tt.Fatal(Twice())\n\t}\n}\n"
	// twice is the function the unbuilt change adds.
	twice = "\n// Twice is twice the value.\nfunc Twice() int { return 2 * Value() }\n"
)

// undone runs the gate's verify over a change that writes source as the fixture's production Go and test as its test,
// on a row of class whose check is undoneCheck, and returns the landing it ran.
func undone(t *testing.T, class, source, test string) (*landing, error) {
	t.Helper()
	fixture := newRowFixture(t, map[string]string{"p/p.go": undoneValue})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Class, document.Items[0].Budget.Nodes = class, 100
		document.Items[0].Steps[0].Verify = undoneCheck
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "the row's check")
	fixture.write("p/p.go", source)
	fixture.write("p/p_test.go", "package p\n\nimport \"testing\"\n\n"+test)
	l := fixture.candidate()
	return l, l.verify(t.Context())
}

// TestTheLandingUndoesItsChangeAndTheCheckFails runs a landed step's check with the change's production Go restored
// and its tests kept: by hand, the lead undid each change and wrote what the check did. A behaviour change whose test
// fails on its assertion without it is caught and says so in the message; one whose test passes without it is refused;
// one whose test does not build without it is said to be no evidence, not counted as a catch; and a consolidation,
// which passes before and after, takes no undo run.
func TestTheLandingUndoesItsChangeAndTheCheckFails(t *testing.T) {
	t.Parallel()
	caught, err := undone(t, "", strings.Replace(undoneValue, "return 1", "return 2", 1), caughtTest)
	if message := caught.message(); err != nil || !strings.Contains(message, undoSection+"\n- with p/p.go restored to ") || !strings.Contains(message, "TestValue") {
		t.Errorf("a change its test catches without it ended as %v, its message:\n%s", err, message)
	}
	if _, err := undone(t, "", undoneValue+"\n// Twice is twice the value.\nfunc Twice() int { return 2 }\n",
		"func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n"); err == nil || !strings.Contains(err.Error(), "proves nothing") {
		t.Errorf("a change whose test passes without it ended as %v, want it refused", err)
	}
	unbuilt, err := undone(t, "", undoneValue+twice, unbuiltTest)
	if err != nil || !strings.Contains(strings.Join(unbuilt.undone, "\n"), "no evidence") {
		t.Errorf("a change whose test does not build without it ended as %v, %q; want it said to be no evidence", err, unbuilt.undone)
	}
	consolidated, err := undone(t, plan.ClassConsolidation, strings.Replace(undoneValue, "// Value is the fixture's value.\n", "// Value is the value.\n", 1),
		"func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n")
	if err != nil || !strings.Contains(strings.Join(consolidated.undone, "\n"), "consolidation") {
		t.Errorf("a consolidation ended as %v, %q; want no undo run", err, consolidated.undone)
	}
}

// TestALandingWithoutUndoEvidenceSaysSo holds a landing whose check does not build with the change's production Go
// restored to saying so where a reader looks: on 2026-11-09 it said so only at the end of its undo section (the
// owner's review). Its message carries the trailer Undo-Evidence: none, which git log counts, and opens its Change
// section with the checks that are no evidence; a change its check catches carries neither.
func TestALandingWithoutUndoEvidenceSaysSo(t *testing.T) {
	t.Parallel()
	const trailer = undoEvidenceTrailer + ": none"
	unbuilt, err := undone(t, "", undoneValue+twice, unbuiltTest)
	unbuilt.composed = "subject\n\nCause:\n- a cause\n\nChange:\n- a change\n"
	if message := unbuilt.message(); err != nil || !strings.Contains(message, "\n"+trailer+"\n") ||
		!strings.Contains(message, "Change:\n- No undo evidence: "+undoneCheck) {
		t.Errorf("a landing whose check does not build without its change ended as %v, its message:\n%s", err, message)
	}
	caught, err := undone(t, "", strings.Replace(undoneValue, "return 1", "return 2", 1), caughtTest)
	if message := caught.message(); err != nil || strings.Contains(message, trailer) || strings.Contains(message, "No undo evidence") {
		t.Errorf("a change its check catches ended as %v, its message:\n%s", err, message)
	}
}
