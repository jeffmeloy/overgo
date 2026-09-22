package closurescan

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
)

// TestRebindDoesNotRepublishUnchangedDocuments holds a rebind to leaving a
// document's identity alone when the declaration only moved: a new line and
// a new owning-file identity, from an edit elsewhere in the file, rebind
// exactly onto the document as it is, while a changed expression rebinds
// onto a new document.
func TestRebindDoesNotRepublishUnchangedDocuments(t *testing.T) {
	previous := rebindCandidate(t, "37", "stable callsites")
	binding, err := previous.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document := rebindDocument(t, previous, binding)
	moved := previous
	moved.Line += 12
	moved.OwnerID = sourceDigest("the file edited elsewhere")
	rebound, matched, reason, err := CompileRebindIndex([]Candidate{moved}).Rebind(document)
	if err != nil {
		t.Fatal(err)
	}
	if !matched || reason != "exact" || rebound.ID != document.ID {
		t.Fatalf("moved declaration rebound = (%t, %s, %s), want exact onto %s", matched, reason, rebound.ID, document.ID)
	}
	changed := moved
	changed.Expression = previous.Expression + " + 0"
	changed.SourceID = sourceDigest("changed expression")
	rebound, matched, reason, err = CompileRebindIndex([]Candidate{changed}).Rebind(document)
	if err != nil {
		t.Fatal(err)
	}
	if !matched || reason != "source" || rebound.ID == document.ID || rebound.Bindings[0].Line != changed.Line {
		t.Fatalf("changed declaration rebound = (%t, %s, %s)", matched, reason, rebound.ID)
	}
}

func TestRebindUnchangedClosureMovesExactBinding(t *testing.T) {
	previous := rebindCandidate(t, "37", "stable callsites")
	current := previous
	current.Line += len(current.Name)
	current.SourceID = sourceDigest("moved source")
	oldBinding, err := previous.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document := rebindDocument(t, previous, oldBinding)
	rebound, matched, reason, err := CompileRebindIndex([]Candidate{current}).Rebind(document)
	if err != nil {
		t.Fatal(err)
	}
	if !matched || reason != "source" || len(rebound.Bindings) != len(document.Bindings) {
		t.Fatalf("rebind = (%t, %s, %d bindings)", matched, reason, len(rebound.Bindings))
	}
	for _, binding := range rebound.Bindings {
		if binding == oldBinding || binding.Line != current.Line || binding.SourceID != current.SourceID {
			t.Fatalf("binding = %+v", binding)
		}
	}
}

func TestRebindDiagnostics(t *testing.T) {
	previous := rebindCandidate(t, "37", "stable callsites")
	binding, err := previous.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document := rebindDocument(t, previous, binding)
	changedValue := previous
	changedValue.Value = "38"
	changedCallsites := previous
	changedCallsites.CallsiteID = sourceDigest("changed callsites")
	for _, test := range []struct {
		candidate Candidate
		reason    string
	}{{changedValue, "value"}, {changedCallsites, "callsite"}} {
		if _, matched, reason, err := CompileRebindIndex([]Candidate{test.candidate}).Rebind(document); err != nil || matched || reason != test.reason {
			t.Fatalf("changed candidate = (%t, %s, %v), want %s", matched, reason, err, test.reason)
		}
	}
}

func TestReviewedRebindAcceptsCallsiteChange(t *testing.T) {
	previous := rebindCandidate(t, "37", "old callsites")
	binding, err := previous.Binding()
	if err != nil {
		t.Fatal(err)
	}
	current := previous
	current.CallsiteID = sourceDigest("reviewed callsites")
	rebound, matched, reason, err := CompileRebindIndex([]Candidate{current}).RebindReviewed(
		rebindDocument(t, previous, binding),
	)
	if err != nil || !matched || reason != "source" || rebound.Bindings[0].CallsiteID != current.CallsiteID {
		t.Fatalf("reviewed rebind = (%t, %s, %v, %+v)", matched, reason, err, rebound.Bindings)
	}
}

func TestRebindMatchesExactBinding(t *testing.T) {
	candidate := rebindCandidate(t, "37", "stable callsites")
	binding, err := candidate.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document := rebindDocument(t, candidate, binding)
	current, matched, reason, err := CompileRebindIndex([]Candidate{candidate}).Rebind(document)
	if err != nil || !matched || reason != "exact" || current.ID != document.ID {
		t.Fatalf("exact rebind = (%s, %t, %s, %v)", current.ID, matched, reason, err)
	}
}

func TestStructuralRebindRejectsAmbiguousMigration(t *testing.T) {
	legacy := rebindCandidate(t, "37", "legacy callsites")
	legacy.StructuralID = ""
	binding, err := legacy.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document := rebindDocument(t, legacy, binding)
	left, right := rebindCandidate(t, "37", "current callsites"), rebindCandidate(t, "37", "current callsites")
	left.StructuralID, right.StructuralID = sourceDigest("left"), sourceDigest("right")
	if _, matched, reason, err := CompileRebindIndex([]Candidate{left, right}).Rebind(document); err != nil || matched || reason != "ambiguous" {
		t.Fatalf("ambiguous migration = (%t, %s, %v)", matched, reason, err)
	}
}

func TestStructuralRebindRecoversReassignedLiteralIdentity(t *testing.T) {
	previous := rebindCandidate(t, "3", "old literal")
	previous.Kind, previous.Name = closureledger.BindingLiteral, "literal.100"
	binding, err := previous.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document := rebindDocument(t, previous, binding)
	collision := previous
	collision.Name, collision.Expression, collision.Value = "literal.90", "0", "0"
	moved := previous
	moved.Name, moved.Line = "literal.200", previous.Line+1
	moved.StructuralID = sourceDigest("moved literal structure")
	moved.SourceID = sourceDigest("moved literal source")
	moved.CallsiteID = sourceDigest("moved literal callsite")
	rebound, matched, reason, err := CompileRebindIndex([]Candidate{collision, moved}).Rebind(document)
	if err != nil || !matched || reason != "source" {
		t.Fatalf("reassigned literal rebind = (%t, %s, %v)", matched, reason, err)
	}
	if got := rebound.Bindings[0]; got.Name != moved.Name || got.StructuralID != moved.StructuralID {
		t.Fatalf("reassigned literal binding = %+v", got)
	}
	rebound, matched, reason, err = CompileRebindIndex([]Candidate{moved}).Rebind(document)
	if err != nil || !matched || reason != "source" || rebound.Bindings[0].StructuralID != moved.StructuralID {
		t.Fatalf("removed literal structure rebind = (%+v, %t, %s, %v)", rebound.Bindings, matched, reason, err)
	}
}

func rebindCandidate(t *testing.T, value, callsites string) Candidate {
	t.Helper()
	name := "PolicyWindow"
	return Candidate{
		Kind: closureledger.BindingConstant, Name: name, File: "internal/policy.go", Package: "internal", Scope: "package",
		Line: len(name), Expression: value, Value: value,
		StructuralID: sourceDigest("policy-window"), SourceID: sourceDigest("original source"), CallsiteID: sourceDigest(callsites),
	}
}

func rebindDocument(t *testing.T, candidate Candidate, binding closureledger.SourceBinding) closureledger.Document {
	t.Helper()
	fixture, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("rebind fixture"))
	if err != nil {
		t.Fatal(err)
	}
	document, err := closureledger.New(
		candidate.Name, json.RawMessage(candidate.Value), closureledger.TierImplementation, closureledger.StatusClosed,
		"Fixed implementation policy.", []closureledger.SourceBinding{binding},
		"Retain while implementation contract holds.", "Implementation contract change.", fixture,
	)
	if err != nil {
		t.Fatal(err)
	}
	return document
}

func sourceDigest(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}
