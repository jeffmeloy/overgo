package landing

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestTheLandingUndoesItsChangeAndTheCheckFails runs a landed step's check with the change's production Go restored
// and its tests kept: by hand, the lead undid each change and wrote what the check did. A behaviour change whose test
// fails on its assertion without it is caught and says so in the message; one whose test passes without it is refused;
// one whose test does not build without it is said to be no evidence, not counted as a catch; and a consolidation,
// which passes before and after, takes no undo run.
func TestTheLandingUndoesItsChangeAndTheCheckFails(t *testing.T) {
	t.Parallel()
	const (
		value = "// Package p is a fixture.\npackage p\n\n// Value is the fixture's value.\nfunc Value() int { return 1 }\n"
		check = "go test ./p -run '^TestValue$' -count=1"
	)
	undone := func(class, source, test string) (*landing, error) {
		t.Helper()
		fixture := newRowFixture(t, map[string]string{"p/p.go": value})
		editPlan(t, fixture, func(document *plan.Plan) {
			document.Items[0].Class, document.Items[0].Budget.Nodes = class, 100
			document.Items[0].Steps[0].Verify = check
		})
		fixture.git("add", "-A")
		fixture.git("commit", "-q", "-m", "the row's check")
		fixture.write("p/p.go", source)
		fixture.write("p/p_test.go", "package p\n\nimport \"testing\"\n\n"+test)
		l := fixture.candidate()
		return l, l.verify(t.Context())
	}

	caught, err := undone("", strings.Replace(value, "return 1", "return 2", 1),
		"func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 2 {\n\t\tt.Fatal(Value())\n\t}\n}\n")
	if message := caught.message(); err != nil || !strings.Contains(message, undoSection+"\n- with p/p.go restored to ") || !strings.Contains(message, "TestValue") {
		t.Errorf("a change its test catches without it ended as %v, its message:\n%s", err, message)
	}
	if _, err := undone("", value+"\n// Twice is twice the value.\nfunc Twice() int { return 2 }\n",
		"func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n"); err == nil || !strings.Contains(err.Error(), "proves nothing") {
		t.Errorf("a change whose test passes without it ended as %v, want it refused", err)
	}
	unbuilt, err := undone("", value+"\n// Twice is twice the value.\nfunc Twice() int { return 2 * Value() }\n",
		"func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Twice() != 2 {\n\t\tt.Fatal(Twice())\n\t}\n}\n")
	if err != nil || !strings.Contains(strings.Join(unbuilt.undone, "\n"), "no evidence") {
		t.Errorf("a change whose test does not build without it ended as %v, %q; want it said to be no evidence", err, unbuilt.undone)
	}
	consolidated, err := undone(plan.ClassConsolidation, strings.Replace(value, "// Value is the fixture's value.\n", "// Value is the value.\n", 1),
		"func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n")
	if err != nil || !strings.Contains(strings.Join(consolidated.undone, "\n"), "consolidation") {
		t.Errorf("a consolidation ended as %v, %q; want no undo run", err, consolidated.undone)
	}
}
