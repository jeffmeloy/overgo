package workflowrecipe

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

type workflowFixture struct {
	bindings Bindings
}

func newWorkflowFixture(t *testing.T) workflowFixture {
	t.Helper()
	return workflowFixture{bindings: Bindings{
		Model:             testutil.ArtifactID(t, artifact.KindModel, "model"),
		Profile:           testutil.ArtifactID(t, artifact.KindProfile, "profile"),
		ProjectionProfile: testutil.ArtifactID(t, artifact.KindProfile, "projection-profile"),
		Tokenizer:         testutil.ArtifactID(t, artifact.KindTokenizer, "tokenizer"),
		Projector:         testutil.ArtifactID(t, artifact.KindProjector, "projector"),
		Dataset:           testutil.ArtifactID(t, artifact.KindDataset, "dataset"),
		Checkpoint:        testutil.ArtifactID(t, artifact.KindCheckpoint, "checkpoint"),
		Adapters: []artifact.ID{
			testutil.ArtifactID(t, artifact.KindAdapter, "adapter-0"),
			testutil.ArtifactID(t, artifact.KindAdapter, "adapter-1"),
		},
	}}
}

func TestWorkflowRecipesCompileCanonicalPlans(t *testing.T) {
	fixture := newWorkflowFixture(t)
	tests := []struct {
		name    string
		build   func() (recipe.Definition, error)
		task    recipe.Task
		steps   int
		support ExecutionSupport
	}{
		{"generation", func() (recipe.Definition, error) {
			return Generation(fixture.bindings, recipe.PlacementHybrid)
		}, recipe.TaskGeneration, 3, ExecutionRuntime},
		{"embedding", func() (recipe.Definition, error) {
			return Embedding(fixture.bindings, recipe.PlacementDevice)
		}, recipe.TaskEmbedding, 4, ExecutionRuntime},
		{"rerank", func() (recipe.Definition, error) {
			return Rerank(fixture.bindings, recipe.PlacementHost)
		}, recipe.TaskRerank, 3, ExecutionRuntime},
		{"projection", func() (recipe.Definition, error) {
			return Projection(fixture.bindings, MediaAudio, recipe.PlacementDevice)
		}, recipe.TaskProjection, 2, ExecutionRuntime},
		{"training", func() (recipe.Definition, error) {
			return Training(fixture.bindings, recipe.PlacementHybrid)
		}, recipe.TaskTraining, 4, ExecutionOrchestration},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			definition, err := test.build()
			if err != nil {
				t.Fatal(err)
			}
			plan, err := Compile(definition)
			if err != nil {
				t.Fatal(err)
			}
			stages := plan.Program.Stages()
			if definition.Task != test.task || len(stages) != test.steps || plan.Support != test.support {
				t.Fatalf("plan = %+v", plan)
			}
			rebuilt, err := test.build()
			if err != nil || rebuilt.ID != definition.ID {
				t.Fatalf("canonical rebuild = (%s, %v)", rebuilt.ID, err)
			}
			positions := make(map[recipe.NodeID]int, len(stages))
			for index, stage := range stages {
				positions[stage.Node.ID] = index
			}
			for _, edge := range definition.Edges {
				if positions[edge.From.Node] >= positions[edge.To.Node] {
					t.Fatalf("steps are not topological: %+v", stages)
				}
			}
		})
	}
}

func TestProjectionMediaContracts(t *testing.T) {
	bindings := newWorkflowFixture(t).bindings
	tests := []struct {
		media   MediaKind
		decode  recipe.ModuleID
		project recipe.ModuleID
		tensor  recipe.DataKind
	}{
		{MediaImage, ModuleDecodeImage, ModuleProjectImage, recipe.DataImageTensor},
		{MediaAudio, ModuleDecodeAudio, ModuleProjectAudio, recipe.DataAudioTensor},
		{MediaVideo, ModuleDecodeVideo, ModuleProjectVideo, recipe.DataVideoTensor},
	}
	for _, test := range tests {
		var priorModules []recipe.ModuleID
		for _, placement := range []recipe.Placement{recipe.PlacementHost, recipe.PlacementDevice, recipe.PlacementHybrid} {
			definition, err := Projection(bindings, test.media, placement)
			if err != nil {
				t.Fatal(err)
			}
			plan, err := Compile(definition)
			if err != nil {
				t.Fatalf("media %d: %v", test.media, err)
			}
			stages := plan.Program.Stages()
			modules := []recipe.ModuleID{stages[0].Module.ID, stages[1].Module.ID}
			if !slices.Equal(modules, []recipe.ModuleID{test.decode, test.project}) ||
				stages[0].Module.Outputs[0].Data != test.tensor || stages[1].Module.Inputs[0].Data != test.tensor {
				t.Fatalf("media %d contract = modules %v, edge %s -> %s", test.media, modules,
					stages[0].Module.Outputs[0].Data, stages[1].Module.Inputs[0].Data)
			}
			if priorModules != nil && !slices.Equal(modules, priorModules) {
				t.Fatalf("media %d modules changed with placement: %v -> %v", test.media, priorModules, modules)
			}
			priorModules = modules
		}
	}
	if _, err := Projection(bindings, 0, recipe.PlacementHost); err == nil {
		t.Fatal("invalid media contract accepted")
	}
}

func TestWorkflowRecipesRequireTaskDependencies(t *testing.T) {
	bindings := newWorkflowFixture(t).bindings
	bindings.Tokenizer = artifact.ID{}
	if _, err := Generation(bindings, recipe.PlacementHost); err == nil {
		t.Fatal("generation without tokenizer accepted")
	}
	bindings = newWorkflowFixture(t).bindings
	bindings.Projector = artifact.ID{}
	if _, err := Projection(bindings, MediaImage, recipe.PlacementHost); err == nil {
		t.Fatal("projection without projector accepted")
	}
	bindings = newWorkflowFixture(t).bindings
	bindings.ProjectionProfile = artifact.ID{}
	if _, err := Projection(bindings, MediaAudio, recipe.PlacementHost); err == nil {
		t.Fatal("audio projection without projection profile accepted")
	}
	if _, err := Projection(bindings, MediaImage, recipe.PlacementHost); err != nil {
		t.Fatalf("image projection rejected without audio profile: %v", err)
	}
	bindings = newWorkflowFixture(t).bindings
	bindings.Dataset = artifact.ID{}
	if _, err := Training(bindings, recipe.PlacementHost); err == nil {
		t.Fatal("training without dataset accepted")
	}
}
