package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// TestInventoryResolvesARelativeRoot holds the inventory to answering the same
// for a relative root as for an absolute one. Registrations record absolute
// locations, so a root matched as given tied no model to its directory and
// reported registered models as unregistered, with commands that would
// register them twice.
func TestInventoryResolvesARelativeRoot(t *testing.T) {
	parent := t.TempDir()
	weights := filepath.Join(parent, "models", "Model", "weights.safetensors")
	if err := os.MkdirAll(filepath.Dir(weights), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(weights, []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(parent)
	directories, err := modelDirectories([]string{"models"})
	if err != nil {
		t.Fatal(err)
	}
	if len(directories) != 1 || !filepath.IsAbs(directories[0].Path) {
		t.Fatalf("directories = %+v, want one absolute path", directories)
	}
	model, err := artifact.IdentifyBytes(artifact.KindModel, []byte("registered"))
	if err != nil {
		t.Fatal(err)
	}
	cells := []ModelValidation{{Model: model, Location: weights}}
	inventory := buildInventory(directories, cells, nil, func(ModelValidation) error { return nil })
	if len(inventory) != 1 || inventory[0].Standing != standingRegistered || len(inventory[0].Models) != 1 {
		t.Fatalf("a registered model under a relative root was not tied to its directory: %+v", inventory)
	}
}

// TestEveryEvidenceEntryHasACurrencyOwner holds every surface a declared task
// can put evidence on to having something that judges its currency or a
// stated exclusion. A surface with neither leaves its models stale however
// often the validators run. Inference is judged by the long-form guard, and
// evaluation cells sit beside a guard cell on the same model; every other
// surface has an acceptance package, or is excluded and says why.
func TestEveryEvidenceEntryHasACurrencyOwner(t *testing.T) {
	t.Parallel()
	for _, task := range []recipe.Task{
		recipe.TaskInference, recipe.TaskGeneration, recipe.TaskEmbedding, recipe.TaskRerank, recipe.TaskProjection,
		recipe.TaskTraining, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq, recipe.TaskSpeech,
		recipe.TaskTranscription, recipe.TaskAlignment, recipe.TaskDiarization, recipe.TaskActivityDetection,
		recipe.TaskAudioConversion, recipe.TaskAudioGeneration, recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskVQA,
	} {
		for _, cell := range kindValidation(task) {
			_, accepted := surfaceAcceptances[cell.Surface]
			_, excluded := excludedSurfaces[cell.Surface]
			guarded := cell.Surface == "inference" || cell.Surface == "evaluation"
			if !accepted && !excluded && !guarded {
				t.Errorf("task %s puts %s evidence on surface %s, which nothing judges and nothing excludes", task, cell.Validation, cell.Surface)
			}
		}
	}
	// A model whose only evidence is on an excluded surface is told why, not
	// that nothing owns it.
	model, err := artifact.IdentifyBytes(artifact.KindModel, []byte("trained"))
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("accepted"))
	if err != nil {
		t.Fatal(err)
	}
	cells := []ModelValidation{{Model: model, Location: "C:/models/Trained/m.safetensors", Validation: "training-proof", Surface: "training", Evidence: evidence}}
	inventory := buildInventory([]modelDirectory{{Path: "C:/models/Trained", Weights: true}}, cells, nil,
		func(ModelValidation) error { return errCellNotJudged })
	if got := inventory[0].Models[0]; got.Standing != standingActivated || got.Currency != excludedSurfaces["training"] {
		t.Fatalf("training-only model = %s %q", got.Standing, got.Currency)
	}
}

// TestInventoryNamesTheProvenFile holds a directory's standing to naming the
// file whose evidence earned it. A directory keeps conversions and superseded
// files beside the one that is proven, and its label alone does not say
// which; a directory with nothing proven names nothing.
func TestInventoryNamesTheProvenFile(t *testing.T) {
	t.Parallel()
	identify := func(text string) artifact.ID {
		id, err := artifact.IdentifyBytes(artifact.KindModel, []byte(text))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	evidence, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("accepted"))
	if err != nil {
		t.Fatal(err)
	}
	superseded, proven, unproven := identify("superseded"), identify("proven"), identify("unproven")
	cells := []ModelValidation{
		{Model: superseded, Location: "C:/models/Model/model-f16.gguf"},
		{Model: proven, Location: "C:/models/Model/model-ropefix.gguf", Validation: guardValidation, Surface: "inference", Evidence: evidence},
		{Model: unproven, Location: "C:/models/Other/model.gguf"},
	}
	directories := []modelDirectory{{Path: "C:/models/Model", Weights: true}, {Path: "C:/models/Other", Weights: true}}
	currency := func(cell ModelValidation) error {
		if cell.Model == proven {
			return errors.New("the long-form record measured another inference surface; run the guard")
		}
		return nil
	}
	inventory := buildInventory(directories, cells, nil, currency)
	if inventory[0].Standing != standingActivated || inventory[0].Proven != "C:/models/Model/model-ropefix.gguf" {
		t.Fatalf("directory %s = %s proven %q", inventory[0].Directory, inventory[0].Standing, inventory[0].Proven)
	}
	if inventory[1].Proven != "" {
		t.Fatalf("a directory with nothing proven names %q", inventory[1].Proven)
	}
}
