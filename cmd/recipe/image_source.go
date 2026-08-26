package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/diffusionimage"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
	"overgo/internal/sensenovarecipe"
)

func resolveImageSource(path string) (capabilitySource, error) {
	routed, err := sensenovarecipe.Recognize(path)
	if err != nil {
		return capabilitySource{}, err
	}
	if routed {
		inventory, err := sensenovarecipe.Inventory(path)
		if err != nil {
			return capabilitySource{}, err
		}
		profile, err := routedlm.InspectFlowProfile(path)
		if err != nil {
			return capabilitySource{}, err
		}
		return capabilitySource{inventory: inventory, define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
			content, err := profile.Content()
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			definition, err := modelrecipe.RoutedImageDefinition(modelID, profile.ID)
			return definition, []artifact.Content{content}, err
		}}, nil
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
	if recognized, err := diffusionimage.Recognize(path); err != nil {
		return capabilitySource{}, err
	} else if recognized {
		inventory, err := imageGenInventory(path)
		return definitionSource(inventory, err, modelrecipe.DiffusionImageDefinition)
	}
	if recognized, err := oscillatorimage.Recognize(path); err != nil {
		return capabilitySource{}, err
	} else if !recognized {
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
			logical, role := directory+"/weights", artifact.ComponentWeights
			if len(weights) > 1 {
				logical, role = fmt.Sprintf("%s/weights-%05d", directory, index), artifact.ComponentWeightsShard
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
		specs[index] = modelartifact.FileSpec{Path: filepath.Join(path, file.path), Name: file.name, Role: file.role}
	}
	return modelartifact.FromFiles(path, specs)
}
