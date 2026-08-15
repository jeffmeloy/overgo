//go:build !windows

package main

import (
	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
)

func imageCapability() capability {
	return capability{
		inventory: imageGenInventory,
		execute: capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, latentimage.EncodedImage](
			"image-gen", oscillatorimage.ValidateRequest,
			capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
		),
		bind: func(_ string, modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
			definition, err := modelrecipe.OscillatorImageDefinition(modelID)
			return definition, nil, err
		},
	}
}
