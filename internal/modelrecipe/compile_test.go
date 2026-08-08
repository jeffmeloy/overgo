package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

func TestInferenceRecipeCompilesExistingModelPlan(t *testing.T) {
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	definition, err := inferenceFixture(modelID, recipe.PlacementHost)
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
	if plan.Recipe.ID != definition.ID || plan.Model.Profile().Family != model.ArchitectureFamilyAttention {
		t.Fatalf("compiled plan = %+v", plan)
	}
	if len(plan.Nodes) != 3 || plan.Decode.Session != DecodeSessionRequest {
		t.Fatalf("compiled runtime program = %+v", plan)
	}
}

func TestRuntimeProgramOwnsCapacityDecodePolicy(t *testing.T) {
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("capacity-model"))
	definition, err := inferenceFixture(modelID, recipe.PlacementHybrid)
	if err != nil {
		t.Fatal(err)
	}
	program, err := Compile(definition, model.Spec{CommonSpec: model.CommonSpec{
		Architecture: "llama", BlockCount: 1,
	}}, model.Weights{Layers: []model.LayerWeights{{}}})
	if err != nil {
		t.Fatal(err)
	}
	if program.Decode.Session != DecodeSessionCapacity || len(program.Nodes) != 3 {
		t.Fatalf("runtime program = %+v", program)
	}
}

func TestIdentityBoundQwen35ProgramOwnsDenseAndRecurrentLayers(t *testing.T) {
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("qwen35-model"))
	profileID, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("qwen35-profile"))
	definitionID, _ := artifact.IdentifyBytes(artifact.KindModelDefinition, []byte("qwen35-definition"))
	definition, err := InferenceWithModelDefinition(
		modelID, profileID, definitionID, recipe.PlacementHybrid,
	)
	if err != nil {
		t.Fatal(err)
	}
	spec := model.Spec{
		CommonSpec:    model.CommonSpec{Architecture: "qwen35", BlockCount: 2},
		AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4},
		RecurrentSpec: model.RecurrentSpec{
			SSMConvKernel: 3, SSMInnerSize: 4, SSMStateSize: 2,
			SSMTimeStepRank: 2, SSMGroupCount: 1, FullAttentionInterval: 2,
		},
	}
	program, err := Compile(definition, spec, model.Weights{
		TokenEmbedding: gguf.TensorInfo{Name: "token_embd.weight"},
		Layers:         []model.LayerWeights{{Recurrent: true}, {}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := program.ValidateServing(); err != nil {
		t.Fatal(err)
	}
	first, _ := program.Model.Layer(0)
	second, _ := program.Model.Layer(1)
	if !first.Recurrent || second.Recurrent ||
		program.Decode.Session != DecodeSessionCapacity {
		t.Fatalf("Qwen3.5 program = %+v", program)
	}
}

func TestRecipeContentPersistsWithoutStorageCoupling(t *testing.T) {
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("persisted-model"))
	definition, err := inferenceFixture(modelID, recipe.PlacementDevice)
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
