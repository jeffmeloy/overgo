//go:build !windows

package mediacapability

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
)

func videoCapability() Capability {
	execute := capabilityruntime.JSONScalar[oscillatorimage.VideoRequest, *oscillatorimage.Model, oscillatorimage.EncodedVideo](
		"video-gen", oscillatorimage.ValidateVideoRequest,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ oscillatorimage.VideoRequest) (*oscillatorimage.Model, error) {
			return oscillatorimage.Load(path)
		}, oscillatorimage.RegisterRuntime,
	)
	return Capability{
		Resolve: resolveDeclaredVideoSource,
		Execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleOscillatorVideoPrepare: execute,
		}.Execute,
	}
}
