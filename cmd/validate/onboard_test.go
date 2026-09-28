package main

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/operation"
)

// TestOneCommandOnboardsADeclaredModel holds validate -onboard to the one
// order that takes a model directory to validated with no hand step between:
// a Hugging Face directory converts to GGUF under the checkpoints, recording
// into the store; the conversion registers, records, replays and activates
// through the library validation; and the long-form guard runs on the same
// file. A run a failed validation stopped is run again on the conversion the
// store now ties to the directory, without converting twice; a GGUF the store
// already ties under another name is onboarded instead of a new conversion; a
// directory with an activated model, or with neither a GGUF nor a config, is
// refused before any step runs. The inventory names this command for a
// directory nothing has registered.
func TestOneCommandOnboardsADeclaredModel(t *testing.T) {
	t.Parallel()
	checkpoints := t.TempDir()
	directory := filepath.Join(t.TempDir(), "Tiny-LM")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{modelConfigFile, "model.safetensors"} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	converted := filepath.Join(checkpoints, conversionDirectory, "Tiny-LM.gguf")
	var ran, validated []string
	var tied []InventoryModel
	var refusal error
	steps := onboarding{
		store: "STORE", checkpoints: checkpoints,
		registered: func(context.Context, string) ([]InventoryModel, error) { return tied, nil },
		run: func(_ context.Context, name string, args []string) error {
			ran = append(ran, name+" "+strings.Join(args, " "))
			return nil
		},
		validate: func(_ context.Context, model, projector string) (operation.Completion, error) {
			validated = append(validated, model+"|"+projector)
			return operation.Completion{}, refusal
		},
	}
	onboard := func(path string) error {
		ran, validated = nil, nil
		return steps.onboard(t.Context(), io.Discard, path)
	}
	if err := onboard(directory); err != nil {
		t.Fatal(err)
	}
	guard, arguments, err := reacquireCommand(ModelValidation{Validation: guardValidation, Location: converted})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"go run ./cmd/hf-gguf-convert -source " + directory + " -output " + converted + " -record STORE",
		guard + " " + strings.Join(arguments, " "),
	}
	if !slices.Equal(ran, want) || !slices.Equal(validated, []string{converted + "|"}) {
		t.Fatalf("onboarding ran %q and validated %q, want %q and %q", ran, validated, want, converted)
	}

	if err := os.MkdirAll(filepath.Dir(converted), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(converted, []byte("converted"), 0o600); err != nil {
		t.Fatal(err)
	}
	tied, refusal = []InventoryModel{{Location: converted, Cells: []ModelValidation{{Location: converted}}}}, errors.New("exact mismatch")
	if err := onboard(directory); err == nil || !strings.Contains(err.Error(), "onboard: validate") || len(ran) != 0 || !slices.Equal(validated, []string{converted + "|"}) {
		t.Fatalf("a rerun after a refused validation: %v, ran %q, validated %q", err, ran, validated)
	}

	if err := os.Remove(converted); err != nil {
		t.Fatal(err)
	}
	earlier := filepath.Join(checkpoints, conversionDirectory, "Tiny-LM-f16.gguf")
	tied = []InventoryModel{{Location: earlier, Cells: []ModelValidation{{Location: earlier}}}}
	if err := onboard(directory); err == nil || len(ran) != 0 || !slices.Equal(validated, []string{earlier + "|"}) {
		t.Fatalf("a registered conversion under another name: %v, ran %q, validated %q", err, ran, validated)
	}

	tied = []InventoryModel{{Location: earlier, Cells: []ModelValidation{{Location: earlier, Kind: "inference"}}}}
	if err := onboard(directory); err == nil || !strings.Contains(err.Error(), "already activated") || len(ran)+len(validated) != 0 {
		t.Fatalf("an activated directory was onboarded again: %v, ran %q, validated %q", err, ran, validated)
	}

	if err := onboard(t.TempDir()); err == nil || len(ran)+len(validated) != 0 {
		t.Fatalf("a directory with no model was onboarded: %v, ran %q, validated %q", err, ran, validated)
	}

	if commands := buildInventory([]modelDirectory{{Path: directory, Weights: true}}, nil, nil, nil)[0].Commands; !slices.Equal(commands, []string{onboardCommand + filepath.ToSlash(directory)}) {
		t.Fatalf("an unregistered directory names %q", commands)
	}
}
