package closurescan

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
)

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
	rebound, changed, err := CompileRebindIndex([]Candidate{current}).Rebind(document)
	if err != nil {
		t.Fatal(err)
	}
	if !changed || len(rebound.Bindings) != len(document.Bindings) {
		t.Fatalf("rebind = (%t, %d bindings)", changed, len(rebound.Bindings))
	}
	for _, binding := range rebound.Bindings {
		if binding == oldBinding || binding.Line != current.Line || binding.SourceID != current.SourceID {
			t.Fatalf("binding = %+v", binding)
		}
	}
}

func TestRebindRefusesChangedValueOrCallsites(t *testing.T) {
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
	for _, candidate := range []Candidate{changedValue, changedCallsites} {
		if _, changed, err := CompileRebindIndex([]Candidate{candidate}).Rebind(document); err != nil || changed {
			t.Fatalf("changed candidate rebound: %t, %v", changed, err)
		}
	}
}

func rebindCandidate(t *testing.T, value, callsites string) Candidate {
	t.Helper()
	name := "PolicyWindow"
	return Candidate{
		Name: name, File: "internal/policy.go", Package: "internal", Scope: "package",
		Line: len(name), Expression: value, Value: value,
		SourceID: sourceDigest("original source"), CallsiteID: sourceDigest(callsites),
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
