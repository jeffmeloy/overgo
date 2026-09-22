package main

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
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
	for _, entry := range buildInventory(directories, cells, recorded, func(ModelValidation) error { return nil }) {
		line := filepath.Base(entry.Directory) + " " + entry.Standing
		for _, tied := range entry.Models {
			line += " " + tied.Link + ":" + tied.Standing
		}
		standings = append(standings, line+" commands="+strings.Join(entry.Commands, ";"))
	}
	want := []string{
		"Converted validated converted-from:validated converted-from:registered commands=go run ./cmd/recipe verify -task TASK -input SUITE.json " + superseded.String(),
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

// TestInventoryValidatedMeansCurrentAtTheSurface holds the inventory to calling
// a model validated only when its guard evidence is current. An accepted
// recipe says a model was validated once; the guard record is keyed to the
// inference code surface, and any change to the closure of that surface
// expires it, so a model whose cells all carry accepted evidence but whose
// guard ran on another surface is activated, keeps the reason with the run
// that lifts it, and keeps the guard's command. A cell without accepted
// evidence outranks the guard question: that model is registered and its guard
// is not asked about. A model that earns no guard is activated too, and says
// that its evidence is unchecked here: nothing is called validated on a claim
// this command cannot check. A directory takes its best model's standing.
func TestInventoryValidatedMeansCurrentAtTheSurface(t *testing.T) {
	t.Parallel()
	identify := func(kind artifact.Kind, text string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(text))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	evidence := identify(artifact.KindRecipe, "accepted")
	current, expired, unverified, image := identify(artifact.KindModel, "current"), identify(artifact.KindModel, "expired"), identify(artifact.KindModel, "unverified"), identify(artifact.KindModel, "image")
	cells := []ModelValidation{
		{Model: current, Location: "C:/models/Current/m.gguf", Validation: guardValidation, Surface: "inference", Evidence: evidence},
		{Model: expired, Location: "C:/models/Expired/m.gguf", Validation: guardValidation, Surface: "inference", Evidence: evidence},
		{Model: expired, Location: "C:/models/Expired/m.gguf", Validation: "bbh", Surface: "evaluation", Evidence: evidence},
		{Model: unverified, Location: "C:/models/Unverified/m.gguf", Validation: guardValidation, Surface: "inference", Evidence: evidence},
		{Model: unverified, Location: "C:/models/Unverified/m.gguf", Validation: "bbh", Surface: "evaluation"},
		{Model: image, Location: "C:/models/Image/m.safetensors", Validation: "image-proof", Surface: "image", Evidence: evidence},
		{Model: expired, Location: "C:/models/Both/old.gguf", Validation: guardValidation, Surface: "inference", Evidence: evidence},
		{Model: current, Location: "C:/models/Both/new.gguf", Validation: guardValidation, Surface: "inference", Evidence: evidence},
	}
	var asked []string
	currency := func(cell ModelValidation) error {
		asked = append(asked, cell.Location)
		switch {
		case cell.Validation == "bbh":
			return errCellNotJudged
		case cell.Validation == "image-proof":
			return errors.New("no complete receipt for the acceptance package at this tree's inputs")
		case strings.Contains(cell.Location, "Expired") || strings.Contains(cell.Location, "old.gguf") || strings.Contains(cell.Location, "Unverified"):
			return errors.New("the long-form record measured another inference surface; run the guard")
		}
		return nil
	}
	directories := []modelDirectory{
		{Path: "C:/models/Current", Weights: true}, {Path: "C:/models/Expired", Weights: true},
		{Path: "C:/models/Unverified", Weights: true}, {Path: "C:/models/Image", Weights: true}, {Path: "C:/models/Both", Weights: true},
	}
	var standings []string
	for _, entry := range buildInventory(directories, cells, nil, currency) {
		line := filepath.Base(entry.Directory) + " " + entry.Standing
		for _, tied := range entry.Models {
			line += " " + tied.Standing
			if tied.Currency != "" {
				line += "(guard refused)"
			}
		}
		standings = append(standings, line+" commands="+strconv.Itoa(len(entry.Commands)))
	}
	want := []string{
		"Both validated activated(guard refused) validated commands=1",
		"Current validated validated commands=0",
		"Expired activated activated(guard refused) commands=1",
		"Image activated activated(guard refused) commands=1",
		"Unverified registered registered commands=1",
	}
	if !slices.Equal(standings, want) {
		t.Fatalf("inventory:\n%s\nwant:\n%s", strings.Join(standings, "\n"), strings.Join(want, "\n"))
	}
	if slices.ContainsFunc(asked, func(location string) bool { return strings.Contains(location, "Unverified") }) {
		t.Errorf("the guard was asked about a model with no accepted evidence: %v", asked)
	}
}
