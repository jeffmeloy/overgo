package modelrecipe

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func prototypeFixture(t *testing.T) ModelPrototype {
	t.Helper()
	return ModelPrototype{
		Architecture:        "adaptive-causal",
		Parent:              testutil.ArtifactID(t, artifact.KindModel, "prototype-parent"),
		ArchitectureProfile: testutil.ArtifactID(t, artifact.KindProfile, "prototype-profile"),
		DerivationPolicy:    testutil.ArtifactID(t, artifact.KindProfile, "prototype-derivation"),
		TrainingObjective:   testutil.ArtifactID(t, artifact.KindRecipe, "prototype-objective"),
		DevelopmentSplit:    testutil.ArtifactID(t, artifact.KindDataset, "prototype-development"),
		PromotionSplit:      testutil.ArtifactID(t, artifact.KindDataset, "prototype-promotion"),
		Hypothesis: PrototypeHypothesis{
			Capability: "long-context retrieval",
			Prediction: "retrieval accuracy improves on the held-out promotion split",
			Falsifier:  "promotion-split retrieval does not beat the incumbent",
		},
		Budget:       PrototypeBudget{GPUMinutes: 90, HostRAMGiB: 32, VRAMGiB: 24, MaxWallNS: 1_000_000_000},
		CodeEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "prototype-code-manifest"),
	}
}

// TestModelPrototypeIdentityAndValidation pins the prototype contract: a
// hypothesis is content-addressed data whose architecture must resolve to a
// compiled Go registration, whose policy and evidence dependencies are exact
// identities, whose promotion split is independent of development, whose
// hypothesis carries a falsifier, and whose code identity is evidence that
// authorizes nothing.
func TestModelPrototypeIdentityAndValidation(t *testing.T) {
	prototype, err := NewModelPrototype(prototypeFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	again, err := NewModelPrototype(prototypeFixture(t))
	if err != nil || again.ID != prototype.ID {
		t.Fatalf("prototype identity is not deterministic: %s vs %s (%v)", again.ID, prototype.ID, err)
	}
	batch, err := prototype.Batch("model-prototype/" + prototype.ID.String())
	if err != nil || len(batch.Contents) != 1 {
		t.Fatalf("prototype batch = (%+v, %v)", batch, err)
	}
	decoded, err := ParseModelPrototype(batch.Contents[0].Data)
	if err != nil || decoded.ID != prototype.ID || decoded.Architecture != prototype.Architecture {
		t.Fatalf("prototype round trip = (%+v, %v)", decoded, err)
	}
	if lineage := prototype.Lineage(); len(lineage) != 7 {
		t.Fatalf("prototype lineage covers %d of its dependencies", len(lineage))
	}

	unregistered := prototypeFixture(t)
	unregistered.Architecture = "not-a-compiled-registration"
	if _, err := NewModelPrototype(unregistered); err == nil ||
		!strings.Contains(err.Error(), "no compiled Go registration") {
		t.Fatalf("unregistered architecture admitted: %v", err)
	}
	dependent := prototypeFixture(t)
	dependent.PromotionSplit = dependent.DevelopmentSplit
	if _, err := NewModelPrototype(dependent); err == nil ||
		!strings.Contains(err.Error(), "independent of the development split") {
		t.Fatalf("dependent promotion split admitted: %v", err)
	}
	unfalsifiable := prototypeFixture(t)
	unfalsifiable.Hypothesis.Falsifier = ""
	if _, err := NewModelPrototype(unfalsifiable); err == nil ||
		!strings.Contains(err.Error(), "falsifier") {
		t.Fatalf("unfalsifiable hypothesis admitted: %v", err)
	}
	unbudgeted := prototypeFixture(t)
	unbudgeted.Budget.GPUMinutes = 0
	if _, err := NewModelPrototype(unbudgeted); err == nil ||
		!strings.Contains(err.Error(), "budget") {
		t.Fatalf("unbudgeted prototype admitted: %v", err)
	}
	executable := prototypeFixture(t)
	executable.CodeEvidence = testutil.ArtifactID(t, artifact.KindRecipe, "prototype-code-as-recipe")
	if _, err := NewModelPrototype(executable); err == nil ||
		!strings.Contains(err.Error(), "cannot authorize execution") {
		t.Fatalf("executable code identity admitted: %v", err)
	}
	wrongParent := prototypeFixture(t)
	wrongParent.Parent = testutil.ArtifactID(t, artifact.KindDataset, "prototype-parent-dataset")
	if _, err := NewModelPrototype(wrongParent); err == nil ||
		!strings.Contains(err.Error(), "parent must be a model identity") {
		t.Fatalf("non-model parent admitted: %v", err)
	}
}
