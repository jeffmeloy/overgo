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

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// onboardingFixture: the checkpoints a Hugging Face directory converts into,
// the conversion it is written to, what the store ties to each directory, and
// steps that record what they run and validate.
type onboardingFixture struct {
	steps          onboarding
	converted      string
	tied           map[string][]InventoryModel
	refusal        error
	ran, validated []string
	retired        []artifact.ID
}

func newOnboardingFixture(t *testing.T) *onboardingFixture {
	t.Helper()
	checkpoints := t.TempDir()
	fixture := &onboardingFixture{
		converted: filepath.Join(checkpoints, conversionDirectory, "Tiny-LM.gguf"),
		tied:      map[string][]InventoryModel{},
	}
	fixture.steps = onboarding{
		store: "STORE", checkpoints: checkpoints,
		registered: func(_ context.Context, directory string) ([]InventoryModel, error) {
			return fixture.tied[directory], nil
		},
		run: func(_ context.Context, name string, args []string) error {
			fixture.ran = append(fixture.ran, name+" "+strings.Join(args, " "))
			return nil
		},
		validate: func(_ context.Context, model, projector string) (operation.Completion, error) {
			fixture.validated = append(fixture.validated, model+"|"+projector)
			return operation.Completion{}, fixture.refusal
		},
		retire: func(_ context.Context, _ string, conversion artifact.ID) error {
			fixture.retired = append(fixture.retired, conversion)
			return nil
		},
	}
	return fixture
}

// huggingFaceDirectory writes a Tiny-LM directory, under its own parent, with
// the config given.
func huggingFaceDirectory(t *testing.T, config string) string {
	t.Helper()
	directory := filepath.Join(t.TempDir(), "Tiny-LM")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{modelConfigFile: config, "model.safetensors": "{}"} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return directory
}

// writeConversion stands in for a conversion written at path.
func writeConversion(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("converted"), 0o600); err != nil {
		t.Fatal(err)
	}
}

func (fixture *onboardingFixture) onboard(t *testing.T, path string) error {
	fixture.ran, fixture.validated, fixture.retired = nil, nil, nil
	return fixture.steps.onboard(t.Context(), io.Discard, path)
}

// TestOnboardingAQwen35DirectoryWritesNoGGUF holds validate -onboard to serving a directory the converter serves as it
// is from its own files: the directory itself registers, records, replays and activates, then its declarations are
// recorded under the model the store holds for its location (model-config), which it knows once it registered, the
// long-form guard runs on it, and no conversion is written. An activated conversion of it is not a refusal but the
// model it takes the place of, retired once the directory is activated and never before. A run a later step stopped,
// the directory activated already, goes on from that step and keeps its goldens.
func TestOnboardingAQwen35DirectoryWritesNoGGUF(t *testing.T) {
	t.Parallel()
	fixture := newOnboardingFixture(t)
	directory := huggingFaceDirectory(t, `{"model_type":"qwen3_5"}`)
	conversion := testutil.ArtifactID(t, artifact.KindModel, "Tiny-LM-f16.gguf")
	earlier := filepath.Join(filepath.Dir(fixture.converted), "Tiny-LM-f16.gguf")
	converted := InventoryModel{Model: conversion, Location: earlier, Cells: []ModelValidation{{Location: earlier, Kind: "inference"}}}
	fixture.tied[directory] = []InventoryModel{converted}
	if err := fixture.onboard(t, directory); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"go run ./cmd/model-config -source " + directory + " -repo STORE " + directory,
		"go run ./cmd/longform -guard -publish -repo STORE -corpus " + guardCorpus + " -budget " + guardBudget + " -model-budget " + guardModelBudget + " " + directory,
	}
	if !slices.Equal(fixture.ran, want) || !slices.Equal(fixture.validated, []string{directory + "|"}) || !slices.Equal(fixture.retired, []artifact.ID{conversion}) {
		t.Fatalf("onboarding ran %q, validated %q and retired %v; want %q, the directory, and its conversion", fixture.ran, fixture.validated, fixture.retired, want)
	}
	if _, err := os.Stat(fixture.converted); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a conversion was written at %s: %v", fixture.converted, err)
	}

	fixture.refusal = errors.New("exact mismatch")
	if err := fixture.onboard(t, directory); err == nil || len(fixture.retired) != 0 {
		t.Errorf("a refused validation: %v, retired %v; want the refusal and nothing retired", err, fixture.retired)
	}
	fixture.tied[directory] = []InventoryModel{converted, {Location: directory, Cells: []ModelValidation{{Location: directory, Kind: "inference"}}}}
	if err := fixture.onboard(t, directory); err != nil || !slices.Equal(fixture.ran, want) || len(fixture.validated) != 0 || !slices.Equal(fixture.retired, []artifact.ID{conversion}) {
		t.Errorf("a run on an activated directory: %v, ran %q, validated %q, retired %v; want the steps after validation and its goldens kept", err, fixture.ran, fixture.validated, fixture.retired)
	}
}

// TestOnboardingAgainRetiresOnlyAConversionStillServing holds validate -onboard, run again on a directory served from
// its own files, to retiring a conversion only for the inference it still serves: the 4B's second onboarding found
// its conversion retired for inference and still activated for training, tried to retire it again and was refused. A
// conversion activated for training alone is left as it is; one activated for inference and training is retired.
func TestOnboardingAgainRetiresOnlyAConversionStillServing(t *testing.T) {
	t.Parallel()
	fixture := newOnboardingFixture(t)
	directory := huggingFaceDirectory(t, `{"model_type":"qwen3_5"}`)
	conversion := testutil.ArtifactID(t, artifact.KindModel, "Tiny-LM-f16.gguf")
	earlier := filepath.Join(filepath.Dir(fixture.converted), "Tiny-LM-f16.gguf")
	served := InventoryModel{Location: directory, Cells: []ModelValidation{{Location: directory, Kind: recipe.TaskInference}}}
	training := InventoryModel{Model: conversion, Location: earlier, Cells: []ModelValidation{{Location: earlier, Kind: recipe.TaskTraining}}}
	fixture.tied[directory] = []InventoryModel{training, served}
	if err := fixture.onboard(t, directory); err != nil || len(fixture.retired) != 0 {
		t.Fatalf("a run again beside a conversion activated for training alone: %v, retired %v; want nothing retired", err, fixture.retired)
	}
	training.Cells = append(training.Cells, ModelValidation{Location: earlier, Kind: recipe.TaskInference})
	fixture.tied[directory] = []InventoryModel{training, served}
	if err := fixture.onboard(t, directory); err != nil || !slices.Equal(fixture.retired, []artifact.ID{conversion}) {
		t.Fatalf("a run again beside a conversion still serving inference: %v, retired %v; want it retired", err, fixture.retired)
	}
}

// TestOneCommandOnboardsADeclaredModel holds validate -onboard to the one
// order that takes a model directory to validated with no hand step between:
// a Hugging Face directory converts to GGUF under the checkpoints, recording
// into the store; the conversion registers, records, replays and activates
// through the library validation; and the long-form guard runs on the same
// file. A directory with an activated model, or with neither a GGUF nor a
// config, is refused before any step runs. The inventory names this command
// for a directory nothing has registered.
func TestOneCommandOnboardsADeclaredModel(t *testing.T) {
	t.Parallel()
	fixture := newOnboardingFixture(t)
	directory := huggingFaceDirectory(t, "{}")
	if err := fixture.onboard(t, directory); err != nil {
		t.Fatal(err)
	}
	// Conversion and the guard both name the one selected store.
	converted := fixture.converted
	want := []string{
		"go run ./cmd/gguf convert -source " + directory + " -output " + converted + " -record STORE",
		"go run ./cmd/longform -guard -publish -repo STORE -corpus " + guardCorpus + " -budget " + guardBudget + " -model-budget " + guardModelBudget + " " + converted,
	}
	if !slices.Equal(fixture.ran, want) || !slices.Equal(fixture.validated, []string{converted + "|"}) {
		t.Fatalf("onboarding ran %q and validated %q, want %q and %q", fixture.ran, fixture.validated, want, converted)
	}

	writeConversion(t, converted)
	fixture.tied[directory] = []InventoryModel{{Location: converted, Cells: []ModelValidation{{Location: converted, Kind: "inference"}}}}
	if err := fixture.onboard(t, directory); err == nil || !strings.Contains(err.Error(), "already activated") || len(fixture.ran)+len(fixture.validated) != 0 {
		t.Fatalf("an activated directory was onboarded again: %v, ran %q, validated %q", err, fixture.ran, fixture.validated)
	}

	if err := fixture.onboard(t, t.TempDir()); err == nil || len(fixture.ran)+len(fixture.validated) != 0 {
		t.Fatalf("a directory with no model was onboarded: %v, ran %q, validated %q", err, fixture.ran, fixture.validated)
	}

	if commands := buildInventory([]modelDirectory{{Path: directory, Weights: true}}, nil, nil, "STORE")[0].Commands; !slices.Equal(commands, []string{onboardCommand + filepath.ToSlash(directory)}) {
		t.Fatalf("an unregistered directory names %q", commands)
	}
}

// TestOnboardingReusesOnlyTheSameSourcesConversion: onboarding reuses only a
// conversion the store ties to the directory. Another directory of the same
// name finds the first one's conversion where its own would go and is
// refused, neither validating nor overwriting it, as is a partial output; a
// run a refused validation stopped reuses its tied conversion without
// converting twice; a conversion tied under another name wins over an
// untied file at the default name; and a directory tied to two conversions
// is refused, naming both.
func TestOnboardingReusesOnlyTheSameSourcesConversion(t *testing.T) {
	t.Parallel()
	fixture := newOnboardingFixture(t)
	first := huggingFaceDirectory(t, `{"model_type":"llama"}`)
	if err := fixture.onboard(t, first); err != nil {
		t.Fatal(err)
	}
	converted := fixture.converted
	writeConversion(t, converted)
	fixture.tied[first] = []InventoryModel{{Location: converted, Cells: []ModelValidation{{Location: converted}}}}

	refusedBeforeAnyStep := func(name string, err error) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), "to no conversion of") || len(fixture.ran)+len(fixture.validated) != 0 {
			t.Fatalf("%s: %v, ran %q, validated %q", name, err, fixture.ran, fixture.validated)
		}
	}
	second := huggingFaceDirectory(t, `{"model_type":"qwen2"}`)
	refusedBeforeAnyStep("another directory of the same name", fixture.onboard(t, second))
	if data, err := os.ReadFile(converted); err != nil || string(data) != "converted" {
		t.Fatalf("the first directory's conversion was disturbed: %q, %v", data, err)
	}

	fixture.refusal = errors.New("exact mismatch")
	if err := fixture.onboard(t, first); err == nil || !strings.Contains(err.Error(), "onboard: validate") || len(fixture.ran) != 0 ||
		!slices.Equal(fixture.validated, []string{converted + "|"}) {
		t.Fatalf("a rerun after a refused validation: %v, ran %q, validated %q", err, fixture.ran, fixture.validated)
	}

	earlier := filepath.Join(filepath.Dir(converted), "Tiny-LM-f16.gguf")
	fixture.tied[first] = []InventoryModel{{Location: earlier, Cells: []ModelValidation{{Location: earlier}}}}
	if err := fixture.onboard(t, first); err == nil || len(fixture.ran) != 0 || !slices.Equal(fixture.validated, []string{earlier + "|"}) {
		t.Fatalf("a registered conversion under another name beside an untied default: %v, ran %q, validated %q", err, fixture.ran, fixture.validated)
	}

	fixture.tied[first] = append(fixture.tied[first], InventoryModel{Location: converted, Cells: []ModelValidation{{Location: converted}}})
	if err := fixture.onboard(t, first); err == nil || !strings.Contains(err.Error(), "2 conversions") || len(fixture.ran)+len(fixture.validated) != 0 {
		t.Fatalf("a directory tied to two conversions: %v, ran %q, validated %q", err, fixture.ran, fixture.validated)
	}

	fixture.tied[first] = nil
	refusedBeforeAnyStep("a partial output nothing ties", fixture.onboard(t, first))
}
