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
	if err := definition.Validate(catalog); err != nil {
		t.Fatal(err)
	}
	plan, err := compileInferenceFixture(definition, model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}, model.Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Recipe.ID != definition.ID || plan.Model.Profile().Family != model.ArchitectureFamilyAttention {
		t.Fatalf("compiled plan = %+v", plan)
	}
	if len(plan.Nodes) != 3 || plan.Decode.Session != DecodeSessionRequest ||
		plan.Residency != recipe.ResidencyHostCache || plan.Identity.Residency != plan.Residency {
		t.Fatalf("compiled runtime program = %+v", plan)
	}
}

func TestRuntimeProgramOwnsCapacityDecodePolicy(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "capacity-model")
	definition, err := inferenceFixture(modelID, recipe.PlacementHybrid, DecodeSessionCapacity)
	if err != nil {
		t.Fatal(err)
	}
	program, err := compileInferenceFixture(definition, model.Spec{CommonSpec: model.CommonSpec{
		Architecture: "llama", BlockCount: 1,
	}}, model.Weights{Layers: []model.LayerWeights{{}}})
	if err != nil {
		t.Fatal(err)
	}
	if program.Decode.Session != DecodeSessionCapacity || len(program.Nodes) != 3 ||
		program.Residency != recipe.ResidencyHybridNative {
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
	program, err := compileInferenceFixture(definition, model.Spec{CommonSpec: model.CommonSpec{
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
	if _, err := compileInferenceFixture(
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
		recipe.ResidencyHybridNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	fixture := modeltest.Qwen35DenseRecurrentPair()
	program, err := compileInferenceFixture(definition, fixture.Spec, fixture.ServingWeights())
	if err != nil {
		t.Fatal(err)
	}
	if err := program.ValidateServing(); err != nil {
		t.Fatal(err)
	}
	first, _ := program.Model.Layer(0)
	second, _ := program.Model.Layer(1)
	if !first.Recurrent || second.Recurrent ||
		program.Decode.Session != DecodeSessionCapacity ||
		program.Identity.Residency != recipe.ResidencyHybridNative {
		t.Fatalf("Qwen3.5 program = %+v", program)
	}
}

func TestCapabilityDefinitionsCompileTypedStages(t *testing.T) {
	tests := []struct {
		task       recipe.Task
		placements []recipe.Placement
		inputs     []recipe.DataKind
		output     recipe.DataKind
		modules    []recipe.ModuleID
	}{
		{recipe.TaskForecast, []recipe.Placement{recipe.PlacementHost}, []recipe.DataKind{recipe.DataTensor}, recipe.DataTensor, []recipe.ModuleID{ModuleForecastSeries}},
		{recipe.TaskTabular, []recipe.Placement{recipe.PlacementHost}, []recipe.DataKind{recipe.DataTensor}, recipe.DataTensor, []recipe.ModuleID{ModuleTabularPredict}},
		{recipe.TaskSeq2Seq, []recipe.Placement{recipe.PlacementHost, recipe.PlacementHost, recipe.PlacementHost}, []recipe.DataKind{recipe.DataTokens}, recipe.DataTokens, []recipe.ModuleID{ModuleSeq2SeqEncode, ModuleSeq2SeqPrepare, ModuleSeq2SeqSelect}},
		{recipe.TaskSpeech, []recipe.Placement{recipe.PlacementHost, recipe.PlacementHost, recipe.PlacementHost}, []recipe.DataKind{recipe.DataText}, recipe.DataAudio, []recipe.ModuleID{ModuleSpeechTokenize, ModuleSpeechGenerate, ModuleSpeechDecode}},
		{recipe.TaskImageGen, []recipe.Placement{recipe.PlacementHost, recipe.PlacementHost, recipe.PlacementHost}, []recipe.DataKind{recipe.DataTensor}, recipe.DataImage, []recipe.ModuleID{ModuleImagePrepare, ModuleImageIntegrate, ModuleImageDecode}},
		{recipe.TaskVQA, []recipe.Placement{recipe.PlacementHost, recipe.PlacementDevice}, []recipe.DataKind{recipe.DataImage, recipe.DataText}, recipe.DataText, []recipe.ModuleID{ModuleVQAPrepare, ModuleVQAGenerate}},
	}
	for _, test := range tests {
		t.Run(string(test.task), func(t *testing.T) {
			modelID := testutil.ArtifactID(t, artifact.KindModel, string(test.task)+"-model")
			definition, err := CapabilityDefinition(test.task, modelID)
			if err != nil {
				t.Fatal(err)
			}
			if definition.Task != test.task || definition.Model != modelID || len(definition.Inputs) != len(test.inputs) || len(definition.Outputs) != 1 || definition.Outputs[0].Data != test.output {
				t.Fatalf("definition = %+v", definition)
			}
			for index, kind := range test.inputs {
				if definition.Inputs[index].Data != kind {
					t.Fatalf("input[%d] = %+v", index, definition.Inputs[index])
				}
			}
			program, err := CompileCapability(definition)
			if err != nil {
				t.Fatal(err)
			}
			stages := program.Stages()
			if len(stages) != len(test.modules) {
				t.Fatalf("stages = %+v", stages)
			}
			for index, module := range test.modules {
				if stages[index].Module.ID != module || stages[index].Node.Placement != test.placements[index] {
					t.Fatalf("stage[%d] = %+v", index, stages[index])
				}
			}
			if _, err := compileInferenceFixture(definition, model.Spec{}, model.Weights{}); err == nil {
				t.Fatal("inference compiler accepted capability recipe")
			}
		})
	}
	if _, err := CapabilityDefinition(recipe.TaskInference, artifact.ID{}); err == nil {
		t.Fatal("inference accepted as capability definition")
	}
	if _, err := CapabilityDefinition(recipe.TaskVideoGen, artifact.ID{}); err == nil {
		t.Fatal("video recipe accepted without an executable adapter")
	}
}

func TestImageCapabilitySupportsHybridPlacement(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "hybrid-image-model")
	definition, err := CapabilityDefinitionAt(recipe.TaskImageGen, modelID, recipe.PlacementHybrid)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	for index, stage := range program.Stages() {
		if stage.Node.Placement != recipe.PlacementHybrid {
			t.Fatalf("stage[%d] placement = %q", index, stage.Node.Placement)
		}
	}
}

func TestCompileCapabilityRejectsInferenceProgram(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "capability-inference-model")
	definition, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileCapability(definition); err == nil {
		t.Fatal("capability compiler accepted inference recipe")
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
