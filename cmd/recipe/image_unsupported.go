//go:build !windows

package main

import (
	"fmt"

	"overgo/internal/capabilityruntime"
	"overgo/internal/diffusionimage"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
)

func imageCapability() capability {
	diffusion := capabilityruntime.JSONScalar[diffusionimage.Request, *diffusionimage.Model, latentimage.EncodedImage](
		"image-gen", diffusionimage.ValidateRequest,
		capabilityruntime.IgnoreInput[diffusionimage.Request](diffusionimage.Load), diffusionimage.RegisterRuntime[*diffusionimage.Model],
	)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, latentimage.EncodedImage](
		"image-gen", oscillatorimage.ValidateRequest,
		capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
	)
	return capability{
		resolve: func(path string) (capabilitySource, error) {
			recognized, err := diffusionimage.Recognize(path)
			if err != nil {
				return capabilitySource{}, err
			}
			if recognized {
				inventory, err := imageGenInventory(path)
				return definitionSource(inventory, err, modelrecipe.DiffusionImageDefinition)
			}
			recognized, err = oscillatorimage.Recognize(path)
			if err != nil {
				return capabilitySource{}, err
			}
			if !recognized {
				return capabilitySource{}, fmt.Errorf("image-gen: artifact has no registered image recipe")
			}
			inventory, err := imageGenInventory(path)
			return definitionSource(inventory, err, modelrecipe.OscillatorImageDefinition)
		},
		execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleDiffusionImagePrepare:  diffusion,
			modelrecipe.ModuleOscillatorImagePrepare: oscillator,
		}.Execute,
	}
}
