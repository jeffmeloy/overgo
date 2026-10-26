package modelrecipe

import (
	"slices"
	"testing"

	"overgo/internal/model"
)

// TestProfileDocumentsBeforeALaterFact holds a profile document written before a later fact of its schema
// (laterProfileFacts) to the provenance it was written with: without the fact, its policy holding it at zero, it
// stands; holding the fact otherwise without its provenance, it is refused; and a document derived now covers it.
func TestProfileDocumentsBeforeALaterFact(t *testing.T) {
	profile, found := model.LookupArchitecture("llama")
	if !found {
		t.Fatal("llama profile absent")
	}
	current, err := CatalogProfileProvenance(profile)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(current, func(item ProfileFactProvenance) bool { return slices.Contains(laterProfileFacts, item.Fact) }) {
		t.Fatal("a profile derived now records no provenance for the later facts")
	}
	if _, err := NewProfileDocument(profile); err != nil {
		t.Fatalf("a profile derived now is refused: %v", err)
	}
	// The provenance a document written before the later facts carries: theirs absent, its catalog derivation made under
	// the schema without them.
	earlier := slices.DeleteFunc(slices.Clone(architectureProfileFactSpecs), func(spec ProfileFactSpec) bool {
		return slices.Contains(laterProfileFacts, spec.Fact)
	})
	derivation, err := catalogProfileDerivation(profile, earlier)
	if err != nil {
		t.Fatal(err)
	}
	var written []ProfileFactProvenance
	for _, spec := range earlier {
		written = append(written, ProfileFactProvenance{
			Fact: spec.Fact, Origin: ProfileFactConfig, SourceField: "architecture_catalog." + string(spec.Fact), DerivationID: derivation.ID,
		})
	}
	if _, err := NewProfileDocumentWithProvenance(profile, written); err != nil {
		t.Fatalf("a document written before the later facts is refused: %v", err)
	}
	claimed := profile
	claimed.Runtime.PromptTensorCores = true
	if _, err := NewProfileDocumentWithProvenance(claimed, written); err == nil {
		t.Fatal("a document holding a later fact without its provenance is accepted")
	}
}
