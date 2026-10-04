package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
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
	inventory := buildInventory(directories, cells, nil, "C:/stores/selected")
	if len(inventory) != 1 || inventory[0].Standing != standingRegistered || len(inventory[0].Models) != 1 {
		t.Fatalf("a registered model under a relative root was not tied to its directory: %+v", inventory)
	}
}

// TestInventoryNamesTheProvenFile holds a directory's standing to naming the
// file whose activation earned it. A directory keeps conversions and
// superseded files beside the one that is activated, and its label alone does
// not say which; a directory with nothing activated names nothing.
func TestInventoryNamesTheProvenFile(t *testing.T) {
	t.Parallel()
	identify := func(text string) artifact.ID {
		id, err := artifact.IdentifyBytes(artifact.KindModel, []byte(text))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	activation, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("accepted"))
	if err != nil {
		t.Fatal(err)
	}
	superseded, proven, unproven := identify("superseded"), identify("proven"), identify("unproven")
	cells := []ModelValidation{
		{Model: superseded, Location: "C:/models/Model/model-f16.gguf"},
		{Model: proven, Location: "C:/models/Model/model-ropefix.gguf", Validation: guardValidation, Recipe: activation},
		{Model: unproven, Location: "C:/models/Other/model.gguf"},
	}
	directories := []modelDirectory{{Path: "C:/models/Model", Weights: true}, {Path: "C:/models/Other", Weights: true}}
	inventory := buildInventory(directories, cells, nil, "C:/stores/selected")
	if inventory[0].Standing != standingActivated || inventory[0].Proven != "C:/models/Model/model-ropefix.gguf" {
		t.Fatalf("directory %s = %s proven %q", inventory[0].Directory, inventory[0].Standing, inventory[0].Proven)
	}
	if inventory[1].Proven != "" {
		t.Fatalf("a directory with nothing proven names %q", inventory[1].Proven)
	}
}
