//go:build !windows

package main

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
)

func videoCapability() capability {
	return capability{
		inventory: func(path string) (modelartifact.Inventory, error) {
			recognized, err := oscillatorimage.Recognize(path)
			if err != nil {
				return modelartifact.Inventory{}, err
			}
			if !recognized {
				return modelartifact.Inventory{}, fmt.Errorf("video-gen: artifact has no registered video recipe")
			}
			return imageGenInventory(path)
		},
		execute: capabilityruntime.JSONScalar[oscillatorimage.VideoRequest, *oscillatorimage.Model, oscillatorimage.EncodedVideo](
			"video-gen", oscillatorimage.ValidateVideoRequest,
			func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ oscillatorimage.VideoRequest) (*oscillatorimage.Model, error) {
				return oscillatorimage.Load(path)
			}, oscillatorimage.RegisterVideoRuntime,
		),
		bind: func(_ string, modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
			definition, err := modelrecipe.OscillatorVideoDefinition(modelID)
			return definition, nil, err
		},
	}
}
