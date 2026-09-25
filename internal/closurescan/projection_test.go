package closurescan

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
)

// TestProjectionLeavesARemovedTwinsSurvivorAlone holds the projection to one
// decision per survivor when a literal is removed beside an identical one in
// its scope. Both decisions fall back by expression onto the survivor: where
// the survivor's own decision stayed put it keeps the survivor and the removed
// one goes unmatched; where both moved, neither proves it owns the survivor
// and both go unmatched. Neither shape refuses as a conflicting claim.
func TestProjectionLeavesARemovedTwinsSurvivorAlone(t *testing.T) {
	literal := func(ordinal string) Candidate {
		structure := sourceDigest("scope ordinal " + ordinal)
		return Candidate{
			Kind: closureledger.BindingLiteral, Name: "literal." + ordinal, File: "internal/turn.go",
			Package: "internal", Scope: "turn", Line: len(ordinal), Expression: "1", Value: "1",
			StructuralID: structure, SourceID: sourceDigest("source " + ordinal), CallsiteID: structure,
		}
	}
	decide := func(candidate Candidate) (closureledger.Document, string) {
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatal(err)
		}
		alias, err := closureledger.ActiveAlias(binding)
		if err != nil {
			t.Fatal(err)
		}
		document := rebindDocument(t, candidate, binding)
		document.Understanding = candidate.Name
		document, err = closureledger.New(
			document.Name, document.Value, document.Tier, document.Status, document.Understanding,
			document.Bindings, document.ClosurePath, document.RerankTrigger, document.Fixture,
		)
		if err != nil {
			t.Fatal(err)
		}
		return document, alias
	}
	project := func(survivor Candidate, previous ...Candidate) Projection {
		documents, aliases := make([]closureledger.Document, len(previous)), map[string]artifact.ID{}
		for index, candidate := range previous {
			document, alias := decide(candidate)
			documents[index], aliases[alias] = document, document.ID
		}
		candidates := []Candidate{survivor}
		projection, err := ProjectActiveClosures(
			CompileRebindIndex(candidates), candidates, documents, aliases, true, true, nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		return projection
	}

	survivor, removed := literal("0"), literal("gone")
	held := project(survivor, survivor, removed)
	if held.Unmatched != 1 || held.Pending[0].Document.Understanding != removed.Name {
		t.Fatalf("held survivor projection = unmatched %d, pending %+v", held.Unmatched, held.Pending)
	}
	for _, document := range held.Resolved {
		if document.Understanding != survivor.Name {
			t.Fatalf("removed twin took the survivor: %+v", document)
		}
	}

	renumbered := literal("1")
	contested := project(survivor, renumbered, removed)
	if contested.Unmatched != 2 || len(contested.Resolved) != 0 {
		t.Fatalf("contested survivor projection = unmatched %d, resolved %d", contested.Unmatched, len(contested.Resolved))
	}
}
