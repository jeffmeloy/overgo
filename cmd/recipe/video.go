//go:build !windows

package main

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
)

func videoCapability() capability {
	execute := capabilityruntime.JSONScalar[oscillatorimage.VideoRequest, *oscillatorimage.Model, oscillatorimage.EncodedVideo](
		"video-gen", oscillatorimage.ValidateVideoRequest,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ oscillatorimage.VideoRequest) (*oscillatorimage.Model, error) {
			return oscillatorimage.Load(path)
		}, oscillatorimage.RegisterRuntime,
	)
	return capability{
		resolve: func(path string) (capabilitySource, error) {
			recognized, err := oscillatorimage.Recognize(path)
			if err != nil {
				return capabilitySource{}, err
			}
			if !recognized {
				return capabilitySource{}, fmt.Errorf("video-gen: artifact has no registered video recipe")
			}
			inventory, err := imageGenInventory(path)
			return definitionSource(inventory, err, modelrecipe.OscillatorVideoDefinition)
		},
		execute: capabilityruntime.Dispatch(
			capabilityruntime.ExecutorBinding{Module: modelrecipe.ModuleOscillatorVideoPrepare, Execute: execute},
		),
	}
}
