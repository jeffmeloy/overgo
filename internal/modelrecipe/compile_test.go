package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

func TestInferenceRecipeCompilesExistingModelPlan(t *testing.T) {
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	definition, err := Inference(modelID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	if err := definition.Validate(Catalog()); err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(definition, model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}, model.Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Recipe.ID != definition.ID || plan.Model.Profile.Family != model.ArchitectureFamilyAttention {
		t.Fatalf("compiled plan = %+v", plan)
	}
	if len(plan.Nodes) != 3 || plan.Decode.Session != DecodeSessionRequest {
		t.Fatalf("compiled runtime program = %+v", plan)
	}
}

func TestRuntimeProgramOwnsCapacityDecodePolicy(t *testing.T) {
	program, err := CompileRuntime(model.Spec{CommonSpec: model.CommonSpec{
		Architecture: "llama", BlockCount: 1,
	}}, model.Weights{Layers: []model.LayerWeights{{}}})
	if err != nil {
		t.Fatal(err)
	}
	if program.Decode.Session != DecodeSessionCapacity || len(program.Nodes) != 3 {
		t.Fatalf("runtime program = %+v", program)
	}
}

func TestRecipeContentPersistsWithoutStorageCoupling(t *testing.T) {
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("persisted-model"))
	definition, err := Inference(modelID, recipe.PlacementDevice)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := Batch("fixture/recipe/publish", definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	content, ok, err := store.Content(context.Background(), definition.ID)
	if err != nil || !ok {
		t.Fatalf("content = (%v, %v)", ok, err)
	}
	want, err := definition.Content()
	if err != nil {
		t.Fatal(err)
	}
	if string(content.Data) != string(want) {
		t.Fatal("persisted recipe content changed")
	}
}
