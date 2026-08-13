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
)

func imageCapability() capability {
	latent := capabilityruntime.JSONScalar[latentimage.Request, *latentimage.Generator, latentimage.Image](
		"image-gen", latentimage.ValidateRequest, latentimage.LoadGenerator, latentimage.RegisterRuntime,
	)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, oscillatorimage.Image](
		"image-gen", oscillatorimage.ValidateRequest,
		capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
	)
	return capability{
		inventory: func(path string) (modelartifact.Inventory, error) {
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
			if programPlacement(program) == recipe.PlacementHybrid {
				return latent(ctx, store, path, modelID, program, raw)
			}
			return oscillator(ctx, store, path, modelID, program, raw)
		},
		definition: func(path string, modelID artifact.ID) (recipe.Definition, error) {
			recognized, err := latentimage.IsPipeline(path)
			if err != nil {
				return recipe.Definition{}, err
			}
			placement := recipe.PlacementHost
			if recognized {
				placement = recipe.PlacementHybrid
			}
			return modelrecipe.CapabilityDefinitionAt(recipe.TaskImageGen, modelID, placement)
		},
	}
}

func programPlacement(program recipe.Program) recipe.Placement {
	nodes := program.Definition().Nodes
	if len(nodes) == 0 {
		return ""
	}
	return nodes[0].Placement
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
