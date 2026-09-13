package main

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

// Interior schedule input for parser fixtures, not a runtime default.
const fixtureEpsilon = 0.001

func TestParseCLI(t *testing.T) {
	// Eight blocks, refined twice. Other values exercise explicit overrides.
	want := cliConfig{
		model: "model.gguf", prompt: "prompt", lora: []string{"adapter.gguf"},
		diffusion: inference.DiffusionOptions{
			MaxLength: 64, Steps: 16, BlockLength: 8,
			Algorithm: inference.DiffusionMargin, Schedule: inference.DiffusionBlock,
			CFGScale: 1.5, AlgorithmTemperature: 0.2,
			Temperature: 0.4, TopK: 12, TopP: 0.8, Seed: 7,
			AddGumbelNoise: true, ShiftLogits: new(false),
		},
	}
	options := want.diffusion
	config, err := parseCLI([]string{
		"-length", strconv.Itoa(options.MaxLength), "-steps", strconv.Itoa(options.Steps),
		"-algorithm", "2", // The public numeric spelling of DiffusionMargin.
		"-block-length", strconv.Itoa(options.BlockLength),
		"-cfg-scale", fmt.Sprint(options.CFGScale), "-alg-temp", fmt.Sprint(options.AlgorithmTemperature),
		"-temp", fmt.Sprint(options.Temperature), "-top-k", strconv.Itoa(options.TopK),
		"-top-p", fmt.Sprint(options.TopP), "-seed", fmt.Sprint(options.Seed),
		"-gumbel", "-shift-logits", "false",
		"-lora", want.lora[0], want.model, want.prompt,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(config, want) {
		t.Fatalf("config = %+v, want %+v", config, want)
	}
}

func TestParseCLIUsesRecipeDefaults(t *testing.T) {
	policy, found, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("runtime policy: found=%t err=%v", found, err)
	}
	config, err := parseCLI([]string{"-eps", fmt.Sprint(fixtureEpsilon), "model.gguf", "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	want := policy.Interactive.Diffusion
	got := config.diffusion
	if got.MaxLength != want.Length || got.Steps != want.Steps || int(got.Algorithm) != want.Algorithm ||
		got.Temperature != want.Temperature || got.TopK != want.TopK || got.TopP != want.TopP ||
		got.Epsilon != float32(fixtureEpsilon) || got.Schedule != inference.DiffusionTimestep {
		t.Fatalf("diffusion defaults = %+v, want %+v", got, want)
	}
}

func TestParseCLIRejectsInvalidFlags(t *testing.T) {
	epsilon := fmt.Sprint(fixtureEpsilon)
	const modelPath, prompt = "model.gguf", "prompt"
	for _, test := range []struct {
		name      string
		arguments []string
		problem   string
	}{
		{"missing prompt", []string{modelPath}, "usage:"},
		{"missing schedule", []string{modelPath, prompt}, "set exactly one"},
		{"invalid shift", []string{"-eps", epsilon, "-shift-logits", "sometimes", modelPath, prompt}, "invalid -shift-logits"},
		{"unsupported preload", []string{"-eps", epsilon, "-preload", "-native-quant", modelPath, prompt}, "flag provided but not defined: -preload"},
		{"empty adapter", []string{"-eps", epsilon, "-lora", "", modelPath, prompt}, "LoRA path is empty"},
		{"conflicting schedules", []string{"-eps", epsilon, "-block-length", "1", modelPath, prompt}, "set exactly one"},
		// Three positions cannot form complete blocks of two.
		{"partial block", []string{"-length", "3", "-steps", "1", "-block-length", "2", modelPath, prompt}, "block length must divide length"},
		{"unknown algorithm", []string{"-eps", epsilon, "-algorithm", strconv.Itoa(int(inference.DiffusionConfidence) + 1), modelPath, prompt}, "numeric option is out of range"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := parseCLI(test.arguments); err == nil || !strings.Contains(err.Error(), test.problem) {
				t.Fatalf("arguments %v: err=%v, want %q", test.arguments, err, test.problem)
			}
		})
	}
}
