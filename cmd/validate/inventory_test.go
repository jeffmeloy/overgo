package main

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestModelInventory holds the inventory to starting from the directories on
// disk and to tying a registered model to its directory by evidence, never by
// a name. A model is tied by its location beneath the directory, or by a
// config file the conversion recorded whose content is the directory's own, so
// a converted file that lives elsewhere still counts and one that merely
// shares the directory's name does not. A model is activated when every cell
// is activated and none is stale; a directory takes
// its best model's standing, and a superseded artifact beside an activated one
// keeps its own command, into the selected store, without lowering the
// directory. A directory with weights and no tied model is unregistered and
// names the command that registers it; one with neither is not a model.
func TestModelInventory(t *testing.T) {
	t.Parallel()
	identify := func(kind artifact.Kind, text string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(text))
		testcheck.NoError(t, err)
		return id
	}
	activation := identify(artifact.KindRecipe, "accepted")
	located, converted, superseded, named := identify(artifact.KindModel, "located"), identify(artifact.KindModel, "converted"), identify(artifact.KindModel, "superseded"), identify(artifact.KindModel, "named")
	cells := []ModelValidation{
		{Model: located, Location: `C:\models\Located\weights.safetensors`, Validation: "guard", Recipe: activation, Stale: "its recipe was replaced"},
		{Model: located, Location: `C:\models\Located\weights.safetensors`, Validation: "bbh", Recipe: activation},
		{Model: converted, Location: `C:\checkpoints\elsewhere.gguf`, Validation: "guard", Recipe: activation},
		{Model: superseded, Location: `C:\checkpoints\elsewhere-old.gguf`},
		{Model: named, Location: `C:\checkpoints\Unregistered-f16.gguf`, Validation: "guard", Recipe: activation},
	}
	directories := []modelDirectory{
		{Path: "C:/models/Unregistered", Weights: true, ModelType: "llama", ConfigDigests: []string{"own-digest"}},
		{Path: "C:/models/Located", Weights: true},
		{Path: "C:/models/Converted", Weights: true, ConfigDigests: []string{"shared-digest", "other"}},
		{Path: "C:/models/docs"},
	}
	recorded := map[artifact.ID][]string{converted: {"shared-digest"}, superseded: {"shared-digest"}, named: {"a-different-directory"}}
	var standings []string
	for _, entry := range buildInventory(directories, cells, recorded, "C:/stores/selected") {
		var line strings.Builder
		line.WriteString(filepath.Base(entry.Directory) + " " + entry.Standing)
		for _, tied := range entry.Models {
			line.WriteString(" " + tied.Link + ":" + tied.Standing)
		}
		standings = append(standings, line.String()+" commands="+strings.Join(entry.Commands, ";"))
	}
	want := []string{
		"Converted activated converted-from:activated converted-from:registered commands=go run ./cmd/recipe verify -task TASK -input SUITE.json " + superseded.String(),
		"Located registered location:registered commands=go run ./cmd/longform -guard -publish -repo C:/stores/selected -corpus " + guardCorpus + " -budget " + guardBudget + " -model-budget " + guardModelBudget + ` C:\models\Located\weights.safetensors`,
		"Unregistered unregistered commands=go run ./cmd/validate -onboard C:/models/Unregistered",
		"docs not-a-model commands=",
	}
	if !slices.Equal(standings, want) {
		t.Fatalf("inventory:\n%s\nwant:\n%s", strings.Join(standings, "\n"), strings.Join(want, "\n"))
	}

	root := t.TempDir()
	write := func(name, content string) { testutil.WriteTextFile(t, root, name, content) }
	write("Model/config.json", `{"model_type": "llama"}`)
	write("Model/shards/part-1.safetensors", "weights")
	write("notes/README.md", "no weights here")
	write("loose.gguf", "a file beside the directories is not a directory")
	read, err := modelDirectories([]string{root})
	testcheck.NoError(t, err)
	if len(read) != 2 || filepath.Base(read[0].Path) != "Model" || !read[0].Weights || read[0].ModelType != "llama" || len(read[0].ConfigDigests) == 0 {
		t.Fatalf("model directory = %+v", read)
	}
	if filepath.Base(read[1].Path) != "notes" || read[1].Weights || read[1].ModelType != "" {
		t.Fatalf("directory without weights = %+v", read[1])
	}
}

// TestATrainedModelIsTiedByWhereItIs: a trained directory holds the config of the model it was trained from, whose content
// ties neither way. Tied, the base would read as the trained model, refusing its admission, and a conversion of the base
// would read as a conversion of these weights, which admitting the directory retires.
func TestATrainedModelIsTiedByWhereItIs(t *testing.T) {
	t.Parallel()
	base := testutil.ArtifactID(t, artifact.KindModel, "base")
	conversion := testutil.ArtifactID(t, artifact.KindModel, "conversion")
	trained := testutil.ArtifactID(t, artifact.KindModel, "trained")
	const shared = "the digest of the config training kept"
	cells := []ModelValidation{
		{Model: base, Location: `C:\models\Base`, Validation: "guard"},
		{Model: conversion, Location: `C:\checkpoints\Base.gguf`, Validation: "guard"},
		{Model: trained, Location: `C:\checkpoints\Tuned`, Trained: true},
	}
	directories := []modelDirectory{
		{Path: "C:/checkpoints/Tuned", Weights: true, ConfigDigests: []string{shared}},
		{Path: "C:/models/Base", Weights: true, ConfigDigests: []string{shared}},
	}
	recorded := map[artifact.ID][]string{base: {shared}, conversion: {shared}, trained: {shared}}
	inventory := buildInventory(directories, cells, recorded, "C:/stores/selected")
	models := func(entry DirectoryInventory) (tied []artifact.ID) {
		for _, model := range entry.Models {
			tied = append(tied, model.Model)
		}
		return tied
	}
	if got := models(inventory[0]); !slices.Equal(got, []artifact.ID{trained}) {
		t.Fatalf("the trained directory is tied to %v, want its own model alone", got)
	}
	if got := models(inventory[1]); !slices.Equal(got, []artifact.ID{base, conversion}) {
		t.Fatalf("the base's directory is tied to %v, want the base and its conversion", got)
	}
}
