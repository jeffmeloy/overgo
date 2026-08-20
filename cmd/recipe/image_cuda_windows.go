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
	"overgo/internal/diffusionimage"
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

func imageCapability() capability {
	routedDirector, routedErr := capabilityruntime.NewModelSessionDirector[sensenovarecipe.GenerationRequest, *sensenovarecipe.Generator, latentimage.EncodedImage](
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
	routed := sessionExecutor(routedDirector, routedErr)
	latentDirector, err := capabilityruntime.NewModelSessionDirector[latentimage.Request, *latentimage.Generator, latentimage.EncodedImage](
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
	latent := sessionExecutor(latentDirector, err)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, latentimage.EncodedImage](
		"image-gen", oscillatorimage.ValidateRequest,
		capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
	)
	diffusionDirector, diffusionErr := capabilityruntime.NewModelSessionDirector[diffusionimage.Request, *diffusionimage.ResidentGenerator, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity,
		diffusionimage.ValidateRequest, diffusionimage.SessionPolicy,
		func(ctx context.Context, _ artifact.Repository, path string, _ recipe.Program, request diffusionimage.Request) (*diffusionimage.ResidentGenerator, error) {
			return diffusionimage.LoadResidentGenerator(ctx, path, request)
		},
		func(ctx context.Context, generator *diffusionimage.ResidentGenerator, request diffusionimage.Request) error {
			return generator.Reset(ctx, request)
		},
		diffusionimage.RegisterRuntime[*diffusionimage.ResidentGenerator],
	)
	diffusion := sessionExecutor(diffusionDirector, diffusionErr)
	return capability{
		resolve: resolveImageSource,
		execute: capabilityruntime.Dispatch(
			capabilityruntime.ExecutorBinding{Module: modelrecipe.ModuleRoutedImagePrepare, Execute: routed},
			capabilityruntime.ExecutorBinding{Module: modelrecipe.ModuleLatentImagePrepare, Execute: latent},
			capabilityruntime.ExecutorBinding{Module: modelrecipe.ModuleOscillatorImagePrepare, Execute: oscillator},
			capabilityruntime.ExecutorBinding{Module: modelrecipe.ModuleDiffusionImagePrepare, Execute: diffusion},
		),
	}
}

func resolveImageSource(path string) (capabilitySource, error) {
	routed, err := sensenovarecipe.Recognize(path)
	if err != nil {
		return capabilitySource{}, err
	}
	if routed {
		inventory, err := sensenovarecipe.Inventory(path)
		return definitionSource(inventory, err, modelrecipe.RoutedImageDefinition)
	}
	latent, err := latentimage.IsPipeline(path)
	if err != nil {
		return capabilitySource{}, err
	}
	if latent {
		inventory, err := latentImageInventory(path)
		return capabilitySource{inventory: inventory, define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
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
		}}, err
	}
	diffusion, err := diffusionimage.Recognize(path)
	if err != nil {
		return capabilitySource{}, err
	}
	if diffusion {
		inventory, err := imageGenInventory(path)
		return definitionSource(inventory, err, modelrecipe.DiffusionImageDefinition)
	}
	oscillator, err := oscillatorimage.Recognize(path)
	if err != nil {
		return capabilitySource{}, err
	}
	if !oscillator {
		return capabilitySource{}, fmt.Errorf("image-gen: artifact has no registered image recipe")
	}
	inventory, err := imageGenInventory(path)
	return definitionSource(inventory, err, modelrecipe.OscillatorImageDefinition)
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
