package landing

import (
	"bytes"
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
// on a row of class whose check is undoneCheck and whose rationale says, where reason is given, why no test through an
// existing boundary is practical; it returns the landing it ran.
func undone(t *testing.T, class, reason, source, test string) (*landing, error) {
	t.Helper()
	fixture := newRowFixture(t, map[string]string{"p/p.go": undoneValue})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Class, document.Items[0].Budget.Nodes = class, 100
		document.Items[0].Steps[0].Verify = undoneCheck
		if reason != "" {
			// The line is spelled out, not read from noBoundaryLine, so the test builds with the landing before it, and is
			// written as go run ./cmd/plan -note writes a note.
			document.Items[0].Steps[0].Rationale = "Why the row exists.\n\nNote: No boundary test: " + reason + "\n"
		}
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "the row's check")
	fixture.write("p/p.go", source)
	fixture.write("p/p_test.go", "package p\n\nimport \"testing\"\n\n"+test)
	l := fixture.candidate()
	// What the landing prints as it runs is kept, to be read back (printed).
	l.Log = &bytes.Buffer{}
	return l, l.verify(t.Context())
}

// printed is what a landing undone ran printed.
func printed(l *landing) string {
	if log, kept := l.Log.(*bytes.Buffer); kept {
		return log.String()
	}
	return ""
}

// TestTheLandingUndoesItsChangeAndTheCheckFails runs a landed step's check with the change's production Go restored
// and its tests kept: by hand, the lead undid each change and wrote what the check did. A behaviour change whose test
// fails on its assertion without it is caught and says so in the message, which carries no word of missing evidence;
// one whose test passes without it is refused; and a consolidation, which passes before and after, takes no undo run.
// A change whose test does not build without it is TestNoUndoEvidenceNeedsABoundaryTestOrAReason's.
func TestTheLandingUndoesItsChangeAndTheCheckFails(t *testing.T) {
	t.Parallel()
	caught, err := undone(t, "", "", strings.Replace(undoneValue, "return 1", "return 2", 1), caughtTest)
	if message := caught.message(); err != nil || !strings.Contains(message, undoSection+"\n- with p/p.go restored to ") || !strings.Contains(message, "TestValue") ||
		strings.Contains(message, undoEvidenceTrailer) || strings.Contains(message, "No undo evidence") || strings.Contains(printed(caught), "has no undo evidence") {
		t.Errorf("a change its test catches without it ended as %v, its message:\n%s\nprinting:\n%s", err, message, printed(caught))
	}
	if _, err := undone(t, "", "", undoneValue+"\n// Twice is twice the value.\nfunc Twice() int { return 2 }\n",
		"func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n"); err == nil || !strings.Contains(err.Error(), "proves nothing") {
		t.Errorf("a change whose test passes without it ended as %v, want it refused", err)
	}
	consolidated, err := undone(t, plan.ClassConsolidation, "", strings.Replace(undoneValue, "// Value is the fixture's value.\n", "// Value is the value.\n", 1),
		"func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n")
	if err != nil || !strings.Contains(strings.Join(consolidated.undone, "\n"), "consolidation") {
		t.Errorf("a consolidation ended as %v, %q; want no undo run", err, consolidated.undone)
	}
}

// TestAMovedTestIsUndoneWhereItMoved holds the undo to a change that moves its test to another file: on 2026-11-11 the
// undo kept the file the change moved it from, so the test was declared in both, the package built in neither, and
// test-reuse's check was refused as no evidence though it failed on its assertion with the old code. The moved test
// runs once in the undone tree and catches the change.
func TestAMovedTestIsUndoneWhereItMoved(t *testing.T) {
	t.Parallel()
	before := "func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n"
	fixture := newRowFixture(t, map[string]string{"p/p.go": undoneValue, "p/old_test.go": "package p\n\nimport \"testing\"\n\n" + before})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Budget.Nodes = 100
		document.Items[0].Steps[0].Verify = undoneCheck
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "the row's check")
	fixture.write("p/p.go", strings.Replace(undoneValue, "return 1", "return 2", 1))
	fixture.git("mv", "p/old_test.go", "p/moved_test.go")
	fixture.write("p/moved_test.go", "package p\n\nimport \"testing\"\n\n"+caughtTest)
	l := fixture.candidate()
	l.Log = &bytes.Buffer{}
	err := l.verify(t.Context())
	if undone := strings.Join(l.undone, "\n"); err != nil || !strings.Contains(undone, "TestValue") || strings.Contains(undone, "before a test asserts") {
		t.Errorf("a change that moves its test ended as %v, its undo:\n%s\nprinting:\n%s", err, undone, printed(l))
	}
}

// TestNoUndoEvidenceNeedsABoundaryTestOrAReason holds a change none of whose checks builds without it to a reason: on
// 2026-11-09 such a landing said its checks were no evidence and landed (the owner's review). Its checks not building
// with its production Go restored, it is refused, the refusal naming the way on that works: a line of its step's
// rationale opening noBoundaryLine, never a widened check, which a landing judged by its base's check does not run
// (fewer-refusals/trim-rules). With that line it lands, its undo
// section giving the reason, its message carrying the trailer Undo-Evidence: none, which git log counts, and opening
// its Change section with the checks that are no evidence.
func TestNoUndoEvidenceNeedsABoundaryTestOrAReason(t *testing.T) {
	t.Parallel()
	if _, err := undone(t, "", "", undoneValue+twice, unbuiltTest); err == nil || !strings.Contains(err.Error(), "No boundary test:") || strings.Contains(err.Error(), "-setverify") {
		t.Errorf("a change whose check does not build without it and whose step gives no reason ended as %v, want it refused naming the note and no widened check", err)
	}
	const reason = "Twice is new, and no caller of Value reads it."
	unbuilt, err := undone(t, "", reason, undoneValue+twice, unbuiltTest)
	unbuilt.composed = "subject\n\nCause:\n- a cause\n\nChange:\n- a change\n"
	if message := unbuilt.message(); err != nil || !strings.Contains(message, "\n"+undoEvidenceTrailer+": none\n") ||
		!strings.Contains(message, "Change:\n- No undo evidence: "+undoneCheck) || !strings.Contains(message, reason) ||
		!strings.Contains(printed(unbuilt), "has no undo evidence: "+undoneCheck) {
		t.Errorf("a change whose check does not build without it and whose step says why ended as %v, its message:\n%s\nprinting:\n%s", err, message, printed(unbuilt))
	}
}
