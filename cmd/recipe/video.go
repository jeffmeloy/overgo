//go:build !windows

package main

import (
	"context"

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
		resolve: resolveDeclaredVideoSource,
		execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleOscillatorVideoPrepare: execute,
		}.Execute,
	}
}
