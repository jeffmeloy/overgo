package modelrecipe

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestPrototypeMaterializationClosure pins the materialization contract: a
// successful trial publishes trained bytes, tensor inventory, the resolved
// model definition bound to the prototype's exact architecture and profile,
// and candidate recipes citing the definition, the trial run, and the
// admission — in one atomic store commit carrying no alias mutation — while
// foreign architecture, foreign profile, a missing run receipt, or a
// non-recipe candidate refuses.
func TestPrototypeMaterializationClosure(t *testing.T) {
	profile, _ := model.LookupArchitecture(definitionArchitecture)
	profile.LayerTopology = model.LayerTopologyCausalPostQKNormSkip
	profileDocument, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	weights := testutil.ArtifactID(t, artifact.KindTensorSet, "trial-weights")
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weights,
	}})
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := modelartifact.NewTensorInventoryDocument(
		manifest.ID, modelartifact.TensorFormatGGUF,
		[]modelartifact.TensorFact{{
			Name: "token_embd.weight", Shape: []uint64{definitionTensorWidth, definitionTensorWidth},
			Storage: "f32", Bytes: definitionTensorBytes,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	document, err := NewModelDefinitionDocument(profileDocument, tensors, definitionSpec())
	if err != nil {
		t.Fatal(err)
	}
	resolved := ResolvedModelDefinition{
		Document: document, Profile: profileDocument, Tensors: tensors, Spec: definitionSpec(),
	}
	inventory := modelartifact.Inventory{
		Manifest: manifest, TensorInventory: tensors,
		Components: []artifact.Descriptor{{ID: weights, Size: definitionTensorBytes}},
	}

	specification := prototypeFixture(t)
	specification.Architecture = definitionArchitecture
	specification.ArchitectureProfile = profileDocument.ID
	prototype, err := NewModelPrototype(specification)
	if err != nil {
		t.Fatal(err)
	}
	trial := PrototypeTrial{
		Prototype: prototype,
		Admission: testutil.ArtifactID(t, artifact.KindEvidence, "materialize-admission"),
		Objective: trainingprogram.ObjectiveTokenPrediction,
	}
	trialRun := testutil.ArtifactID(t, artifact.KindRun, "materialize-run")
	candidate := testutil.ArtifactID(t, artifact.KindRecipe, "materialize-candidate-recipe")

	batch, err := MaterializePrototypeTrial(trial, resolved, inventory, trialRun, []artifact.ID{candidate})
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Aliases) != 0 {
		t.Fatalf("materialization moved an alias: %+v", batch.Aliases)
	}
	cited := map[artifact.ID]bool{}
	for _, edge := range batch.Lineage {
		if edge.Child == candidate {
			cited[edge.Parent] = true
		}
	}
	if !cited[document.ID] || !cited[trialRun] || !cited[trial.Admission] {
		t.Fatalf("candidate recipe lost its citations: %+v", batch.Lineage)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatalf("materialization batch is not atomically committable: %v", err)
	}

	foreignArchitecture := trial
	foreignArchitecture.Prototype.Architecture = "adaptive-causal"
	if _, err := MaterializePrototypeTrial(foreignArchitecture, resolved, inventory, trialRun, []artifact.ID{candidate}); err == nil ||
		!strings.Contains(err.Error(), "exact architecture") {
		t.Fatalf("foreign architecture materialized: %v", err)
	}
	foreignProfile := trial
	foreignProfile.Prototype.ArchitectureProfile = testutil.ArtifactID(t, artifact.KindProfile, "foreign-profile")
	if _, err := MaterializePrototypeTrial(foreignProfile, resolved, inventory, trialRun, []artifact.ID{candidate}); err == nil ||
		!strings.Contains(err.Error(), "architecture profile") {
		t.Fatalf("foreign profile materialized: %v", err)
	}
	if _, err := MaterializePrototypeTrial(trial, resolved, inventory, trial.Admission, []artifact.ID{candidate}); err == nil ||
		!strings.Contains(err.Error(), "run receipt") {
		t.Fatalf("missing run receipt materialized: %v", err)
	}
	if _, err := MaterializePrototypeTrial(trial, resolved, inventory, trialRun, []artifact.ID{trial.Admission}); err == nil ||
		!strings.Contains(err.Error(), "exact recipe identities") {
		t.Fatalf("non-recipe candidate materialized: %v", err)
	}
}
