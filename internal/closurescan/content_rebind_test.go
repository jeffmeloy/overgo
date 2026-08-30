package closurescan

import (
	"testing"

	"overgo/internal/closureledger"
)

// TestContentMatchedRebind pins the offset-shift recovery: a document
// whose structural match failed rebinds onto the one candidate with the
// same kind, package, file, scope, constant name, and exact value whose
// alias no active decision holds, carrying the reviewed closure text
// unchanged. An already-claimed successor is not a match, two free
// matching candidates stay ambiguous, a value change refuses, and a
// multi-binding document is out of scope.
func TestContentMatchedRebind(t *testing.T) {
	previous := rebindCandidate(t, "37", "old callsites")
	binding, err := previous.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document := rebindDocument(t, previous, binding)
	successor := previous
	successor.Line += 3
	successor.Expression = "flags.Arg(0)"
	successor.StructuralID = sourceDigest("shifted structure")
	successor.CallsiteID = sourceDigest("shifted callsites")
	successor.SourceID = sourceDigest("shifted source")
	if _, matched, _, err := CompileRebindIndex([]Candidate{successor}).Rebind(document); err != nil || matched {
		t.Fatalf("structural rebind unexpectedly matched the shifted successor: %v", err)
	}

	none := func(string) bool { return false }
	rebound, matched, reason, err := ContentMatchedRebind(document, []Candidate{successor}, none)
	if err != nil || !matched || reason != "content" {
		t.Fatalf("content rebind = (%t, %s, %v)", matched, reason, err)
	}
	if rebound.Bindings[0].Line != successor.Line ||
		rebound.Understanding != document.Understanding || rebound.Tier != document.Tier ||
		string(rebound.Value) != string(document.Value) {
		t.Fatalf("rebound closure lost its reviewed content: %+v", rebound)
	}

	successorBinding, err := successor.Binding()
	if err != nil {
		t.Fatal(err)
	}
	successorAlias, err := closureledger.ActiveAlias(successorBinding)
	if err != nil {
		t.Fatal(err)
	}
	if _, matched, reason, err := ContentMatchedRebind(
		document, []Candidate{successor},
		func(alias string) bool { return alias == successorAlias },
	); err != nil || matched || reason != "declaration" {
		t.Fatalf("claimed successor rebound: (%t, %s, %v)", matched, reason, err)
	}

	twin := successor
	twin.Line += 7
	twin.StructuralID = sourceDigest("twin structure")
	twin.CallsiteID = sourceDigest("twin callsites")
	if _, matched, reason, err := ContentMatchedRebind(
		document, []Candidate{successor, twin}, none,
	); err != nil || matched || reason != "ambiguous" {
		t.Fatalf("two free successors rebound: (%t, %s, %v)", matched, reason, err)
	}

	changed := successor
	changed.Value = "38"
	if _, matched, reason, err := ContentMatchedRebind(
		document, []Candidate{changed}, none,
	); err != nil || matched || reason != "declaration" {
		t.Fatalf("changed value rebound: (%t, %s, %v)", matched, reason, err)
	}

	multi := document
	multi.Bindings = append(append([]closureledger.SourceBinding(nil), document.Bindings...), binding)
	if _, matched, reason, err := ContentMatchedRebind(multi, []Candidate{successor}, none); err != nil ||
		matched || reason != "bindings" {
		t.Fatalf("multi-binding document rebound: (%t, %s, %v)", matched, reason, err)
	}
}
