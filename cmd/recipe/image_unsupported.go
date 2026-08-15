//go:build !windows

package main

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/diffusionimage"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
)

func imageCapability() capability {
	diffusion := capabilityruntime.JSONScalar[diffusionimage.Request, *diffusionimage.Model, latentimage.EncodedImage](
		"image-gen", diffusionimage.ValidateRequest,
		capabilityruntime.IgnoreInput[diffusionimage.Request](diffusionimage.Load), diffusionimage.RegisterRuntime,
	)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, latentimage.EncodedImage](
		"image-gen", oscillatorimage.ValidateRequest,
		capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
	)
	return capability{
		inventory: imageGenInventory,
		execute: func(ctx context.Context, store artifact.Repository, path string, modelID artifact.ID, program recipe.Program, raw string) (any, error) {
			switch imageProgramModule(program) {
			case modelrecipe.ModuleDiffusionImagePrepare:
				return diffusion(ctx, store, path, modelID, program, raw)
			case modelrecipe.ModuleOscillatorImagePrepare:
				return oscillator(ctx, store, path, modelID, program, raw)
			default:
				return nil, fmt.Errorf("image-gen: compiled recipe has no registered operator")
			}
		},
		bind: func(path string, modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
			recognized, err := diffusionimage.Recognize(path)
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			if recognized {
				definition, err := modelrecipe.DiffusionImageDefinition(modelID)
				return definition, nil, err
			}
			recognized, err = oscillatorimage.Recognize(path)
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			if !recognized {
				return recipe.Definition{}, nil, fmt.Errorf("image-gen: artifact has no registered image recipe")
			}
			definition, err := modelrecipe.OscillatorImageDefinition(modelID)
			return definition, nil, err
		},
	}
}
