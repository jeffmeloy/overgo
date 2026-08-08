package closureledger

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// Pins the import-path defect found by the first magic-ledger import: a
// generic map-marshal orders fields alphabetically, the codec's canonical
// form is struct order, and Parse rejects the mismatch. Normalize must admit
// any field order and emit bytes Parse accepts.
func TestNormalizeAdmitsAnyFieldOrderAndEmitsCanonical(t *testing.T) {
	owner := testutil.ArtifactID(t, artifact.KindFile, "owner-surface")
	fixture := testutil.ArtifactID(t, artifact.KindFile, "pinning-fixture")
	alphabetical := []byte(`{"closure_path":"derive from the measured envelope","name":"ExampleMagic",` +
		`"owner_surfaces":["` + owner.String() + `"],"pinning_fixture":"` + fixture.String() + `",` +
		`"rerank_trigger":"re-evaluate on owner change","status":"open","tier":"derivation-blocked",` +
		`"value":8,"version":1}`)

	if _, err := Parse(alphabetical); err == nil {
		t.Fatal("alphabetical field order parsed as canonical; this test no longer pins anything")
	}
	document, canonical, err := Normalize(alphabetical)
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if document.Name != "ExampleMagic" || document.Tier != TierDerivationBlocked {
		t.Fatalf("normalize decoded wrong document: %+v", document)
	}
	reparsed, err := Parse(canonical)
	if err != nil {
		t.Fatalf("canonical bytes rejected by Parse: %v", err)
	}
	if reparsed.ID != document.ID {
		t.Fatal("identity differs between Normalize and Parse of the same canonical bytes")
	}
	again, canonicalAgain, err := Normalize(canonical)
	if err != nil || !bytes.Equal(canonical, canonicalAgain) || again.ID != document.ID {
		t.Fatal("Normalize is not idempotent on canonical input")
	}
}
