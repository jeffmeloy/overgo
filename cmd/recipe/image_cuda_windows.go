//go:build windows

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
	"overgo/internal/sensenovarecipe"
)

const (
	imageDevice          = "cuda:0"
	imageSessionCapacity = 1
)

func residentExecutor[Input, Model, Output any](
	cache *capabilityruntime.ScalarSessionCache[Input, Model, Output],
	err error,
) capabilityruntime.Executor {
	if err == nil {
		return cache.Executor()
	}
	return func(context.Context, artifact.Repository, string, artifact.ID, recipe.Program, string) (any, error) {
		return nil, err
	}
}

func imageCapability() capability {
	routedCache, routedErr := capabilityruntime.NewScalarSessionCache[sensenovarecipe.GenerationRequest, *sensenovarecipe.Generator, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity,
		sensenovarecipe.ValidateGenerationRequest, sensenovarecipe.GenerationSessionPolicy,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ sensenovarecipe.GenerationRequest) (*sensenovarecipe.Generator, error) {
			return sensenovarecipe.LoadGenerator(path)
		},
		func(ctx context.Context, generator *sensenovarecipe.Generator, request sensenovarecipe.GenerationRequest) error {
			return generator.Reset(ctx, request)
		},
		sensenovarecipe.RegisterRuntime,
	)
	routed := residentExecutor(routedCache, routedErr)
	latentCache, err := capabilityruntime.NewScalarSessionCache[latentimage.Request, *latentimage.Generator, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity,
		latentimage.ValidateRequest, latentimage.SessionPolicy,
		func(ctx context.Context, store artifact.Repository, path string, program recipe.Program, request latentimage.Request) (*latentimage.Generator, error) {
			profileID, ok := program.Definition().Dependency(recipe.DependencyProfile, 0)
			if !ok {
				return nil, fmt.Errorf("image-gen: compiled recipe has no profile")
			}
			profile, err := latentimage.ReadProfile(ctx, store, profileID)
			if err != nil {
				return nil, err
			}
			return latentimage.LoadGenerator(ctx, path, profile, request)
		},
		func(ctx context.Context, generator *latentimage.Generator, request latentimage.Request) error {
			return generator.Reset(ctx, request)
		},
		latentimage.RegisterRuntime,
	)
	latent := residentExecutor(latentCache, err)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, latentimage.EncodedImage](
		"image-gen", oscillatorimage.ValidateRequest,
		capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
	)
	return capability{
		inventory: func(path string) (modelartifact.Inventory, error) {
			routedModel, err := sensenovarecipe.Recognize(path)
			if err != nil {
				return modelartifact.Inventory{}, err
			}
			if routedModel {
				return sensenovarecipe.Inventory(path)
			}
			recognized, err := latentimage.IsPipeline(path)
			if err != nil {
				return modelartifact.Inventory{}, err
			}
			if recognized {
				return latentImageInventory(path)
			}
			return imageGenInventory(path)
		},
		execute: func(ctx context.Context, store artifact.Repository, path string, modelID artifact.ID, program recipe.Program, raw string) (any, error) {
			switch imageProgramModule(program) {
			case modelrecipe.ModuleRoutedImagePrepare:
				return routed(ctx, store, path, modelID, program, raw)
			case modelrecipe.ModuleLatentImagePrepare:
				return latent(ctx, store, path, modelID, program, raw)
			case modelrecipe.ModuleOscillatorImagePrepare:
				return oscillator(ctx, store, path, modelID, program, raw)
			default:
				return nil, fmt.Errorf("image-gen: compiled recipe has no registered operator")
			}
		},
		bind: func(path string, modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
			routedModel, err := sensenovarecipe.Recognize(path)
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			if routedModel {
				definition, err := modelrecipe.RoutedImageDefinition(modelID)
				return definition, nil, err
			}
			recognized, err := latentimage.IsPipeline(path)
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			if recognized {
				profile, err := latentimage.ResolveProfile(path)
				if err != nil {
					return recipe.Definition{}, nil, err
				}
				content, err := profile.Content()
				if err != nil {
					return recipe.Definition{}, nil, err
				}
				definition, err := modelrecipe.LatentImageDefinition(modelID, profile.ID)
				return definition, []artifact.Content{content}, err
			}
			definition, err := modelrecipe.OscillatorImageDefinition(modelID)
			return definition, nil, err
		},
	}
}

func imageProgramModule(program recipe.Program) recipe.ModuleID {
	stages := program.Stages()
	if len(stages) == 0 {
		return ""
	}
	return stages[0].Module.ID
}

func latentImageInventory(path string) (modelartifact.Inventory, error) {
	files := []struct {
		path string
		name string
		role artifact.ComponentRole
	}{
		{"model_index.json", "pipeline/config", artifact.ComponentConfig},
		{"scheduler/scheduler_config.json", "scheduler/config", artifact.ComponentConfig},
		{"text_encoder/config.json", "text_encoder/config", artifact.ComponentConfig},
		{"tokenizer/tokenizer.json", "tokenizer", artifact.ComponentTokenizer},
		{"tokenizer/tokenizer_config.json", "tokenizer/config", artifact.ComponentConfig},
		{"transformer/config.json", "transformer/config", artifact.ComponentConfig},
		{"vae/config.json", "vae/config", artifact.ComponentConfig},
	}
	for _, directory := range []string{"text_encoder", "transformer", "vae"} {
		entries, err := os.ReadDir(filepath.Join(path, directory))
		if err != nil {
			return modelartifact.Inventory{}, err
		}
		var weights []string
		indexName := ""
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors") {
				weights = append(weights, entry.Name())
			}
			if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors.index.json") {
				indexName = entry.Name()
			}
		}
		sort.Strings(weights)
		if len(weights) == 0 {
			return modelartifact.Inventory{}, fmt.Errorf("latent image inventory: no %s weights", directory)
		}
		if len(weights) > 1 && indexName == "" {
			return modelartifact.Inventory{}, fmt.Errorf("latent image inventory: %s shards have no index", directory)
		}
		if indexName != "" {
			files = append(files, struct {
				path string
				name string
				role artifact.ComponentRole
			}{filepath.Join(directory, indexName), directory + "/shard-index", artifact.ComponentShardIndex})
		}
		for index, name := range weights {
			logical := directory + "/weights"
			role := artifact.ComponentWeights
			if len(weights) > 1 {
				logical = fmt.Sprintf("%s/weights-%05d", directory, index)
				role = artifact.ComponentWeightsShard
			}
			files = append(files, struct {
				path string
				name string
				role artifact.ComponentRole
			}{filepath.Join(directory, name), logical, role})
		}
	}
	specs := make([]modelartifact.FileSpec, len(files))
	for index, file := range files {
		specs[index] = modelartifact.FileSpec{
			Path: filepath.Join(path, file.path), Name: file.name, Role: file.role,
		}
	}
	return modelartifact.FromFiles(path, specs)
}
