package workflowrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

type workflowFixture struct {
	bindings Bindings
}

func newWorkflowFixture(t *testing.T) workflowFixture {
	t.Helper()
	identify := func(kind artifact.Kind, value string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(value))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	return workflowFixture{bindings: Bindings{
		Model:      identify(artifact.KindModel, "model"),
		Profile:    identify(artifact.KindProfile, "profile"),
		Tokenizer:  identify(artifact.KindTokenizer, "tokenizer"),
		Projector:  identify(artifact.KindProjector, "projector"),
		Dataset:    identify(artifact.KindDataset, "dataset"),
		Checkpoint: identify(artifact.KindCheckpoint, "checkpoint"),
		Adapters: []artifact.ID{
			identify(artifact.KindAdapter, "adapter-0"),
			identify(artifact.KindAdapter, "adapter-1"),
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
			if definition.Task != test.task || len(plan.Steps) != test.steps || plan.Support != test.support {
				t.Fatalf("plan = %+v", plan)
			}
			rebuilt, err := test.build()
			if err != nil || rebuilt.ID != definition.ID {
				t.Fatalf("canonical rebuild = (%s, %v)", rebuilt.ID, err)
			}
			positions := make(map[recipe.NodeID]int, len(plan.Steps))
			for index, step := range plan.Steps {
				positions[step.ID] = index
			}
			for _, edge := range definition.Edges {
				if positions[edge.From.Node] >= positions[edge.To.Node] {
					t.Fatalf("steps are not topological: %+v", plan.Steps)
				}
			}
		})
	}
}

func TestProjectionMediaContracts(t *testing.T) {
	bindings := newWorkflowFixture(t).bindings
	for _, media := range []MediaKind{MediaImage, MediaAudio, MediaVideo} {
		definition, err := Projection(bindings, media, recipe.PlacementHost)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Compile(definition); err != nil {
			t.Fatalf("media %d: %v", media, err)
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
	bindings.Dataset = artifact.ID{}
	if _, err := Training(bindings, recipe.PlacementHost); err == nil {
		t.Fatal("training without dataset accepted")
	}
}
