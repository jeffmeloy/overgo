package main

import (
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/controllertrain"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/scratchmodel"
	"overgo/internal/scratchmodeltest"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

// TestControllerRoutePublishesCheckpoint trains a scratch controller on the
// device from a small controller corpus through the controller route, and
// holds the route to publishing it as a checkpoint recorded in the store
// that starts from the scratch construction the corpus derives.
func TestControllerRoutePublishesCheckpoint(t *testing.T) {
	cudatest.Require(t)
	spec := controllerCorpusSpec()
	path := filepath.Join(t.TempDir(), "corpus.json")
	if err := jsonfile.Write(path, spec, 0o600); err != nil {
		t.Fatal(err)
	}
	const seed = 17
	checkpoint, _ := requireRoutePublishes(t, "controller", "-corpus", path, "-seed", fmt.Sprint(seed))
	corpus, err := controllertrain.Compile(spec)
	if err != nil {
		t.Fatal(err)
	}
	construction, err := scratchmodel.Compile(scratchmodel.CorpusFacts{Documents: corpus.TrainingDocuments(), Seed: seed, Steps: routeTestSteps}, scratchmodeltest.Profile(t))
	if err != nil {
		t.Fatal(err)
	}
	scratch := construction.Authority()
	if checkpoint.Model != scratch.InitializedModel() || checkpoint.Dataset != scratch.Dataset() || checkpoint.Split != scratch.Split() ||
		checkpoint.Program != construction.Program().ID() ||
		!slices.Contains(checkpoint.Lineage, trainingprogram.LineageParent{Artifact: scratch.ID(), Relation: artifact.RelationDependsOn}) {
		t.Fatalf("checkpoint of %s over %s/%s under %s with lineage %v, want scratch construction %s's model %s over %s/%s under %s",
			checkpoint.Model, checkpoint.Dataset, checkpoint.Split, checkpoint.Program, checkpoint.Lineage,
			scratch.ID(), scratch.InitializedModel(), scratch.Dataset(), scratch.Split(), construction.Program().ID())
	}
}

// controllerCorpusSpec is the smallest controller corpus: one training and
// one held-out record for the component and the training-workflow action of
// every modality.
func controllerCorpusSpec() controllertrain.Spec {
	actions := []controllertrain.Action{
		{Scope: controllertrain.ScopeComponent, Task: recipe.TaskGeneration, Modality: controllertrain.ModalityText, Module: workflowrecipe.ModuleGenerate},
		{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityImage, Module: workflowrecipe.ModuleProjectImage},
		{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityAudio, Module: workflowrecipe.ModuleProjectAudio},
		{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityVideo, Module: workflowrecipe.ModuleProjectVideo},
	}
	for _, modality := range []controllertrain.Modality{
		controllertrain.ModalityText, controllertrain.ModalityImage, controllertrain.ModalityAudio, controllertrain.ModalityVideo,
	} {
		actions = append(actions, controllertrain.Action{
			Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: modality, Module: workflowrecipe.ModuleBatchDataset,
		})
	}
	source := controllertrain.GitSource{Commit: "0123456789abcdef0123456789abcdef01234567", Path: "internal/workflowrecipe/catalog.go"}
	// The scratch model's vocabulary is its training split's characters, so
	// every record's prompt carries all of them.
	const prompt = "abcdefghijklmnopqrstuvwxyz 0123456789"
	var spec controllertrain.Spec
	for index, action := range actions {
		record := func(split string) controllertrain.Record {
			id := fmt.Sprintf("%s-%d", split, index)
			return controllertrain.Record{ID: id, Group: id, Source: source, Prompt: prompt, Action: action}
		}
		spec.Train, spec.Holdout = append(spec.Train, record("train")), append(spec.Holdout, record("holdout"))
	}
	return spec
}
