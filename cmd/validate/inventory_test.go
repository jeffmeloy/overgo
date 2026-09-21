package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestModelInventory holds the inventory to starting from the directories on
// disk and to tying a registered model to its directory by evidence, never by
// a name. A model is tied by its location beneath the directory, or by a
// config file the conversion recorded whose content is the directory's own, so
// a converted file that lives elsewhere still counts and one that merely
// shares the directory's name does not. A model is validated when every cell
// carries accepted evidence and none is stale; a directory takes its best
// model's standing, and a superseded artifact beside a validated one keeps its
// own command without lowering the directory. A directory with weights and no
// tied model is unregistered and names the command that registers it; one with
// neither is not a model.
func TestModelInventory(t *testing.T) {
	t.Parallel()
	identify := func(kind artifact.Kind, text string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(text))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	evidence := identify(artifact.KindRecipe, "accepted")
	located, converted, superseded, named := identify(artifact.KindModel, "located"), identify(artifact.KindModel, "converted"), identify(artifact.KindModel, "superseded"), identify(artifact.KindModel, "named")
	cells := []ModelValidation{
		{Model: located, Location: `C:\models\Located\weights.safetensors`, Validation: "guard", Surface: "inference", Evidence: evidence, Stale: "surface moved"},
		{Model: located, Location: `C:\models\Located\weights.safetensors`, Validation: "bbh", Surface: "evaluation", Evidence: evidence},
		{Model: converted, Location: `C:\checkpoints\elsewhere.gguf`, Validation: "guard", Surface: "inference", Evidence: evidence},
		{Model: superseded, Location: `C:\checkpoints\elsewhere-old.gguf`},
		{Model: named, Location: `C:\checkpoints\Unregistered-f16.gguf`, Validation: "guard", Surface: "inference", Evidence: evidence},
	}
	directories := []modelDirectory{
		{Path: "C:/models/Unregistered", Weights: true, ModelType: "llama", ConfigDigests: []string{"own-digest"}},
		{Path: "C:/models/Located", Weights: true},
		{Path: "C:/models/Converted", Weights: true, ConfigDigests: []string{"shared-digest", "other"}},
		{Path: "C:/models/docs"},
	}
	recorded := map[artifact.ID][]string{converted: {"shared-digest"}, superseded: {"shared-digest"}, named: {"a-different-directory"}}
	var standings []string
	for _, entry := range buildInventory(directories, cells, recorded) {
		line := filepath.Base(entry.Directory) + " " + entry.Standing
		for _, tied := range entry.Models {
			line += " " + tied.Link + ":" + tied.Standing
		}
		standings = append(standings, line+" commands="+strings.Join(entry.Commands, ";"))
	}
	want := []string{
		"Converted validated converted-from:validated converted-from:registered commands=go run ./cmd/recipe verify " + superseded.String(),
		"Located registered location:registered commands=go run ./cmd/longform -guard -publish -corpus " + guardCorpus + " -budget " + guardBudget + " -model-budget " + guardModelBudget + ` C:\models\Located\weights.safetensors`,
		"Unregistered unregistered commands=go run ./cmd/recipe spec -root C:/models/ -output Unregistered-registration.json Unregistered;go run ./cmd/recipe register -repo STORE -root C:/models/ -spec Unregistered-registration.json",
		"docs not-a-model commands=",
	}
	if !slices.Equal(standings, want) {
		t.Fatalf("inventory:\n%s\nwant:\n%s", strings.Join(standings, "\n"), strings.Join(want, "\n"))
	}

	root := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("Model/config.json", `{"model_type": "llama"}`)
	write("Model/shards/part-1.safetensors", "weights")
	write("notes/README.md", "no weights here")
	write("loose.gguf", "a file beside the directories is not a directory")
	read, err := modelDirectories([]string{root})
	if err != nil {
		t.Fatal(err)
	}
	if len(read) != 2 || filepath.Base(read[0].Path) != "Model" || !read[0].Weights || read[0].ModelType != "llama" || len(read[0].ConfigDigests) == 0 {
		t.Fatalf("model directory = %+v", read)
	}
	if filepath.Base(read[1].Path) != "notes" || read[1].Weights || read[1].ModelType != "" {
		t.Fatalf("directory without weights = %+v", read[1])
	}
}
