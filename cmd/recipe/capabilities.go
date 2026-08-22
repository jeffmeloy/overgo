package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/seq2seq"
	"overgo/internal/seriesforecast"
	"overgo/internal/speechsynth"
	"overgo/internal/tabularicl"
	"overgo/internal/thoughtbank"
)

type capability struct {
	resolve func(string) (capabilitySource, error)
	execute capabilityruntime.Executor
}

type capabilitySource struct {
	inventory modelartifact.Inventory
	related   []modelartifact.Inventory
	define    func(artifact.ID) (recipe.Definition, []artifact.Content, error)
}

func definitionSource(
	inventory modelartifact.Inventory,
	err error,
	define func(artifact.ID) (recipe.Definition, error),
) (capabilitySource, error) {
	if err != nil {
		return capabilitySource{}, err
	}
	return capabilitySource{inventory: inventory, define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
		definition, err := define(modelID)
		return definition, nil, err
	}}, nil
}

func inventoryCapability(
	inventory func(string) (modelartifact.Inventory, error),
	execute capabilityruntime.Executor,
) capability {
	return capability{
		resolve: func(path string) (capabilitySource, error) {
			resolved, err := inventory(path)
			return capabilitySource{inventory: resolved}, err
		},
		execute: execute,
	}
}

func sessionExecutor[Input, Model, Output any](
	director *capabilityruntime.ModelSessionDirector[Input, Model, Output],
	err error,
) capabilityruntime.Executor {
	if err == nil {
		return director.Executor()
	}
	return func(context.Context, artifact.Repository, string, artifact.ID, recipe.Program, string) (any, error) {
		return nil, err
	}
}

func thoughtBankCapability() capability {
	director, err := capabilityruntime.NewModelSessionDirector[thoughtbank.GenerateRequest, *thoughtbank.Generator, thoughtbank.Generation](
		"generation", "host", recipe.SessionRequestCapacity,
		thoughtbank.ValidateGenerateRequest, thoughtbank.GenerationSessionPolicy,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ thoughtbank.GenerateRequest) (*thoughtbank.Generator, error) {
			return thoughtbank.LoadGenerator(path)
		},
		func(context.Context, *thoughtbank.Generator, thoughtbank.GenerateRequest) error { return nil },
		thoughtbank.RegisterRuntime,
	)
	return capability{
		resolve: func(path string) (capabilitySource, error) {
			inventory, inventoryErr := thoughtBankInventory(path)
			return definitionSource(inventory, inventoryErr, func(modelID artifact.ID) (recipe.Definition, error) {
				return modelrecipe.CapabilityDefinition(recipe.TaskGeneration, modelID)
			})
		},
		execute: sessionExecutor(director, err),
	}
}

func projectionCapability(projectorPath string) capability {
	return capability{resolve: func(modelPath string) (capabilitySource, error) {
		file, err := gguf.Open(modelPath)
		if err != nil {
			return capabilitySource{}, err
		}
		modelInventory, inventoryErr := modelartifact.FromGGUF(file, artifact.KindModel)
		inventoryErr = errors.Join(inventoryErr, file.Close())
		if inventoryErr != nil {
			return capabilitySource{}, inventoryErr
		}
		projectorInventory, media, err := projector.InspectProjection(context.Background(), projectorPath)
		if err != nil {
			return capabilitySource{}, err
		}
		return capabilitySource{
			inventory: modelInventory,
			related:   []modelartifact.Inventory{projectorInventory},
			define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
				definition, err := modelrecipe.ProjectionDefinition(modelID, projectorInventory.Manifest.ID, media...)
				return definition, nil, err
			},
		}, nil
	}}
}

var capabilities = map[recipe.Task]capability{
	recipe.TaskGeneration: thoughtBankCapability(),
	recipe.TaskForecast: inventoryCapability(modelartifact.FromHFPath, capabilityruntime.JSONScalar[[]float32, *seriesforecast.Model, []float32](
		"forecast", seriesforecast.ValidateRequest,
		capabilityruntime.IgnoreInput[[]float32](seriesforecast.Load), seriesforecast.RegisterRuntime)),
	recipe.TaskTabular: inventoryCapability(tabularInventory, capabilityruntime.JSONScalar[tabularicl.Request, *tabularicl.Model, tabularicl.Prediction](
		"tabular", tabularicl.ValidateRequest,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, request tabularicl.Request) (*tabularicl.Model, error) {
			return tabularicl.LoadTask(path, request.Task)
		}, tabularicl.RegisterRuntime)),
	recipe.TaskSeq2Seq: inventoryCapability(modelartifact.FromHFPath, capabilityruntime.JSONScalar[seq2seq.GenerateRequest, *seq2seq.Generator, string](
		"seq2seq", seq2seq.ValidateGenerateRequest,
		capabilityruntime.IgnoreInput[seq2seq.GenerateRequest](seq2seq.LoadGenerator), seq2seq.RegisterRuntime)),
	recipe.TaskSpeech: inventoryCapability(speechInventory, capabilityruntime.JSONScalar[speechsynth.SynthesisRequest, *speechsynth.Synthesizer, speechsynth.Audio](
		"speech", speechsynth.ValidateSynthesisRequest,
		capabilityruntime.IgnoreInput[speechsynth.SynthesisRequest](speechsynth.LoadSynthesizer), speechsynth.RegisterRuntime)),
	recipe.TaskImageGen: imageCapability(),
	recipe.TaskVideoGen: videoCapability(),
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

func thoughtBankInventory(path string) (modelartifact.Inventory, error) {
	return modelartifact.FromFiles(path, []modelartifact.FileSpec{
		{Path: filepath.Join(path, "model.pt"), Name: "weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(path, "tokenizer.json"), Name: "tokenizer", Role: artifact.ComponentTokenizer},
		{Path: filepath.Join(path, "generation_config.json"), Name: "generation", Role: artifact.ComponentConfig},
	})
}
