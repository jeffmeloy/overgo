package finding

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestFindingRoundTripAndNormalizeAnyOrder(t *testing.T) {
	owner := testutil.ArtifactID(t, artifact.KindRecipe, "series-recipe")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "advisory-one")
	document, err := New(
		"confirmed regression in series x", SeverityMedium, StatusOpen,
		[]artifact.ID{owner}, []artifact.ID{evidence},
		"diagnose via phase deltas; close with fix landed.",
		"advisories stop raising across two consecutive windows after the fix.",
	)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := document.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(canonical)
	if err != nil || parsed.ID != document.ID {
		t.Fatalf("round trip failed: %v", err)
	}
	// Alphabetical re-marshal (the importer's resolved form) must be
	// admitted by Normalize and rejected by strict Parse -- the codec owns
	// canonical order.
	alphabetical := []byte(`{"closure_path":"diagnose via phase deltas; close with fix landed.",` +
		`"evidence":["` + evidence.String() + `"],"failable_check":"advisories stop raising across two consecutive windows after the fix.",` +
		`"owner_surfaces":["` + owner.String() + `"],"severity":"medium","status":"open",` +
		`"title":"confirmed regression in series x","version":1}`)
	if _, err := Parse(alphabetical); err == nil {
		t.Fatal("alphabetical order parsed as canonical")
	}
	normalized, normalBytes, err := Normalize(alphabetical)
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if normalized.ID != document.ID || !bytes.Equal(normalBytes, canonical) {
		t.Fatal("normalize did not converge to the canonical form")
	}
	if _, err := New("", SeverityMedium, StatusOpen, []artifact.ID{owner}, []artifact.ID{evidence}, "x", "y"); err == nil {
		t.Fatal("empty title accepted")
	}
	if _, err := New("t", "urgent", StatusOpen, []artifact.ID{owner}, []artifact.ID{evidence}, "x", "y"); err == nil {
		t.Fatal("invalid severity accepted")
	}
}
