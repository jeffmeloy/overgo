package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/seq2seq"
	"overgo/internal/seriesforecast"
	"overgo/internal/speechsynth"
	"overgo/internal/tabularicl"
)

type capability struct {
	inventory func(string) (modelartifact.Inventory, error)
	execute   capabilityruntime.Executor
	bind      func(string, artifact.ID) (recipe.Definition, []artifact.Content, error)
}

var capabilities = map[recipe.Task]capability{
	recipe.TaskForecast: {inventory: modelartifact.FromHFPath, execute: capabilityruntime.JSONScalar[[]float32, *seriesforecast.Model, []float32](
		"forecast", seriesforecast.ValidateRequest,
		capabilityruntime.IgnoreInput[[]float32](seriesforecast.Load), seriesforecast.RegisterRuntime)},
	recipe.TaskTabular: {inventory: tabularInventory, execute: capabilityruntime.JSONScalar[tabularicl.Request, *tabularicl.Model, tabularicl.Prediction](
		"tabular", tabularicl.ValidateRequest,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, request tabularicl.Request) (*tabularicl.Model, error) {
			return tabularicl.LoadTask(path, request.Task)
		}, tabularicl.RegisterRuntime)},
	recipe.TaskSeq2Seq: {inventory: modelartifact.FromHFPath, execute: capabilityruntime.JSONScalar[seq2seq.GenerateRequest, *seq2seq.Generator, string](
		"seq2seq", seq2seq.ValidateGenerateRequest,
		capabilityruntime.IgnoreInput[seq2seq.GenerateRequest](seq2seq.LoadGenerator), seq2seq.RegisterRuntime)},
	recipe.TaskSpeech: {inventory: speechInventory, execute: capabilityruntime.JSONScalar[speechsynth.SynthesisRequest, *speechsynth.Synthesizer, speechsynth.Audio](
		"speech", speechsynth.ValidateSynthesisRequest,
		capabilityruntime.IgnoreInput[speechsynth.SynthesisRequest](speechsynth.LoadSynthesizer), speechsynth.RegisterRuntime)},
	recipe.TaskImageGen: imageCapability(),
	recipe.TaskVideoGen: videoCapability(),
	recipe.TaskVQA:      {inventory: modelartifact.FromHFPath},
}

func safetensorsInventory(context, path, config string, companions ...modelartifact.FileSpec) (modelartifact.Inventory, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	weights := ""
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors") {
			continue
		}
		if weights != "" {
			return modelartifact.Inventory{}, fmt.Errorf("%s inventory: multiple safetensors files in %s", context, path)
		}
		weights = entry.Name()
	}
	if weights == "" {
		return modelartifact.Inventory{}, fmt.Errorf("%s inventory: no safetensors weights in %s", context, path)
	}
	specs := []modelartifact.FileSpec{
		{Path: filepath.Join(path, config), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(path, weights), Name: "weights", Role: artifact.ComponentWeights},
	}
	for _, companion := range companions {
		companion.Path = filepath.Join(path, companion.Path)
		specs = append(specs, companion)
	}
	return modelartifact.FromFiles(path, specs)
}

func speechInventory(path string) (modelartifact.Inventory, error) {
	return safetensorsInventory("speech", path, "pockettts_config.json",
		modelartifact.FileSpec{Path: "tokenizer.model", Name: "tokenizer", Role: artifact.ComponentTokenizer})
}

func imageGenInventory(path string) (modelartifact.Inventory, error) {
	return safetensorsInventory("image-gen", path, "config.json")
}

func tabularInventory(path string) (modelartifact.Inventory, error) {
	var specs []modelartifact.FileSpec
	for _, head := range tabularicl.Tasks() {
		specs = append(specs,
			modelartifact.FileSpec{Path: filepath.Join(path, head, "config.json"), Name: head + "/config", Role: artifact.ComponentConfig},
			modelartifact.FileSpec{Path: filepath.Join(path, head, "model.safetensors"), Name: head + "/weights", Role: artifact.ComponentWeights},
		)
	}
	return modelartifact.FromFiles(path, specs)
}
