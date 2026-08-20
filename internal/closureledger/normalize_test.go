package closureledger

import (
	"bytes"
	"encoding/json"
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
	binding, err := json.Marshal(testSourceBinding(owner))
	if err != nil {
		t.Fatal(err)
	}
	alphabetical := []byte(`{"closure_path":"derive from the measured envelope","name":"ExampleMagic",` +
		`"bindings":[` + string(binding) + `],"pinning_fixture":"` + fixture.String() + `",` +
		`"rerank_trigger":"re-evaluate on owner change","status":"open","tier":"derivation-blocked",` +
		`"understanding":"unresolved owner policy","value":8,"version":3}`)

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

func TestClosureLedgerCurrentContractOnly(t *testing.T) {
	owner := testutil.ArtifactID(t, artifact.KindFile, "legacy-owner")
	fixture := testutil.ArtifactID(t, artifact.KindFile, "legacy-fixture")
	for _, version := range []string{"1", "2"} {
		legacy := []byte(`{"version":` + version + `,"name":"LegacyMagic","value":8,"tier":"derivation-blocked",` +
			`"status":"open","owner_surfaces":["` + owner.String() + `"],` +
			`"closure_path":"derive","rerank_trigger":"source change",` +
			`"pinning_fixture":"` + fixture.String() + `"}`)
		if _, err := Parse(legacy); err == nil {
			t.Fatalf("v%s document parsed", version)
		}
		if _, _, err := Normalize(legacy); err == nil {
			t.Fatalf("v%s document normalized", version)
		}
	}
}
