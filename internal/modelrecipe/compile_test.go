package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestInferenceRecipeCompilesExistingModelPlan(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "model")
	definition, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if err := definition.Validate(Catalog()); err != nil {
		t.Fatal(err)
	}
	plan, err := CompileInference(definition, model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}, model.Weights{})
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
	modelID := testutil.ArtifactID(t, artifact.KindModel, "capacity-model")
	definition, err := inferenceFixture(modelID, recipe.PlacementHybrid, DecodeSessionCapacity)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileInference(definition, model.Spec{CommonSpec: model.CommonSpec{
		Architecture: "llama", BlockCount: 1,
	}}, model.Weights{Layers: []model.LayerWeights{{}}})
	if err != nil {
		t.Fatal(err)
	}
	if program.Decode.Session != DecodeSessionCapacity || len(program.Nodes) != 3 {
		t.Fatalf("runtime program = %+v", program)
	}
}

func TestRecipeDecodeSessionPolicyIsAuthoritative(t *testing.T) {
	const fixtureLayerCount = 1
	modelID := testutil.ArtifactID(t, artifact.KindModel, "request-capable-model")
	definition, err := inferenceFixture(modelID, recipe.PlacementHybrid, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileInference(definition, model.Spec{CommonSpec: model.CommonSpec{
		Architecture: "llama", BlockCount: fixtureLayerCount,
	}}, model.Weights{Layers: []model.LayerWeights{{}}})
	if err != nil {
		t.Fatal(err)
	}
	if program.Decode.Session != DecodeSessionRequest {
		t.Fatalf("decode session = %q, need recipe request policy", program.Decode.Session)
	}
}

func TestCapacityDecodePolicyRequiresCompatibleModel(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "request-only-model")
	definition, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionCapacity)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileInference(
		definition,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}},
		model.Weights{},
	); err == nil {
		t.Fatal("capacity policy accepted a model without capacity cache support")
	}
}

func TestIdentityBoundQwen35ProgramOwnsDenseAndRecurrentLayers(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "qwen35-model")
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "qwen35-profile")
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "qwen35-definition")
	definition, err := InferenceWithModelDefinition(
		modelID, profileID, definitionID, recipe.PlacementHybrid, DecodeSessionCapacity,
	)
	if err != nil {
		t.Fatal(err)
	}
	fixture := modeltest.Qwen35DenseRecurrentPair()
	program, err := CompileInference(definition, fixture.Spec, fixture.ServingWeights())
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

func TestCompileCapabilityResolvesForecastStage(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "forecast-model")
	definition, err := ForecastDefinition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	stages := program.Stages()
	if len(stages) != 1 || stages[0].Node.Module != ModuleForecastSeries ||
		stages[0].Module.ID != ModuleForecastSeries {
		t.Fatalf("forecast stages = %+v", stages)
	}
}

func TestTabularDefinitionValidatesAgainstCatalog(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "tabular-model")
	definition, err := TabularDefinition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	if err := definition.Validate(Catalog()); err != nil {
		t.Fatal(err)
	}
	if definition.Task != recipe.TaskTabular || definition.Model != modelID {
		t.Fatalf("definition = %+v", definition)
	}
	if len(definition.Nodes) != 1 || definition.Nodes[0].Module != ModuleTabularPredict ||
		definition.Nodes[0].Placement != recipe.PlacementHost {
		t.Fatalf("nodes = %+v", definition.Nodes)
	}
	if len(definition.Inputs) != 1 || definition.Inputs[0].Data != recipe.DataTensor ||
		len(definition.Outputs) != 1 || definition.Outputs[0].Data != recipe.DataTensor {
		t.Fatalf("ports = %+v / %+v", definition.Inputs, definition.Outputs)
	}
	// Inference compiler must refuse the non-token task.
	if _, err := CompileInference(definition, model.Spec{}, model.Weights{}); err == nil {
		t.Fatal("inference compiler accepted a tabular recipe")
	}
}

func TestSeq2SeqDefinitionValidatesAgainstCatalog(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "seq2seq-model")
	definition, err := Seq2SeqDefinition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	if err := definition.Validate(Catalog()); err != nil {
		t.Fatal(err)
	}
	if definition.Task != recipe.TaskSeq2Seq || definition.Model != modelID {
		t.Fatalf("definition = %+v", definition)
	}
	if len(definition.Nodes) != 1 || definition.Nodes[0].Module != ModuleSeq2SeqGenerate ||
		definition.Nodes[0].Placement != recipe.PlacementHost {
		t.Fatalf("nodes = %+v", definition.Nodes)
	}
	if len(definition.Inputs) != 1 || definition.Inputs[0].Data != recipe.DataTokens ||
		len(definition.Outputs) != 1 || definition.Outputs[0].Data != recipe.DataTokens {
		t.Fatalf("ports = %+v / %+v", definition.Inputs, definition.Outputs)
	}
	// Inference compiler must refuse the non-inference task.
	if _, err := CompileInference(definition, model.Spec{}, model.Weights{}); err == nil {
		t.Fatal("inference compiler accepted a seq2seq recipe")
	}
}

func TestSpeechDefinitionValidatesAgainstCatalog(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "speech-model")
	definition, err := SpeechDefinition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	if err := definition.Validate(Catalog()); err != nil {
		t.Fatal(err)
	}
	if definition.Task != recipe.TaskSpeech || definition.Model != modelID {
		t.Fatalf("definition = %+v", definition)
	}
	if len(definition.Nodes) != 1 || definition.Nodes[0].Module != ModuleSpeechSynthesize ||
		definition.Nodes[0].Placement != recipe.PlacementHost {
		t.Fatalf("nodes = %+v", definition.Nodes)
	}
	if len(definition.Inputs) != 1 || definition.Inputs[0].Data != recipe.DataText ||
		len(definition.Outputs) != 1 || definition.Outputs[0].Data != recipe.DataAudio {
		t.Fatalf("ports = %+v / %+v", definition.Inputs, definition.Outputs)
	}
	// Inference compiler must refuse the non-inference task.
	if _, err := CompileInference(definition, model.Spec{}, model.Weights{}); err == nil {
		t.Fatal("inference compiler accepted a speech recipe")
	}
}

func TestImageGenDefinitionValidatesAgainstCatalog(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "imagegen-model")
	definition, err := ImageGenDefinition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	if err := definition.Validate(Catalog()); err != nil {
		t.Fatal(err)
	}
	if definition.Task != recipe.TaskImageGen || definition.Model != modelID {
		t.Fatalf("definition = %+v", definition)
	}
	if len(definition.Nodes) != 1 || definition.Nodes[0].Module != ModuleImageGenerate ||
		definition.Nodes[0].Placement != recipe.PlacementHost {
		t.Fatalf("nodes = %+v", definition.Nodes)
	}
	if len(definition.Inputs) != 1 || definition.Inputs[0].Data != recipe.DataTensor ||
		len(definition.Outputs) != 1 || definition.Outputs[0].Data != recipe.DataImage {
		t.Fatalf("ports = %+v / %+v", definition.Inputs, definition.Outputs)
	}
	// Inference compiler must refuse the non-inference task.
	if _, err := CompileInference(definition, model.Spec{}, model.Weights{}); err == nil {
		t.Fatal("inference compiler accepted an image-gen recipe")
	}
}

func TestRecipeContentPersistsWithoutStorageCoupling(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "persisted-model")
	definition, err := inferenceFixture(modelID, recipe.PlacementDevice, DecodeSessionRequest)
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
