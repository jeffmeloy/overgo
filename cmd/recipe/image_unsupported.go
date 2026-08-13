//go:build !windows

package main

import (
	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
)

func imageCapability() capability {
	return capability{
		inventory: imageGenInventory,
		execute: capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, oscillatorimage.Image](
			"image-gen", oscillatorimage.ValidateRequest,
			capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
		),
		definition: func(_ string, modelID artifact.ID) (recipe.Definition, error) {
			return modelrecipe.OscillatorImageDefinition(modelID)
		},
	}
}
