//go:build !windows

package mediacapability

import (
	"context"
	"errors"

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

// reconstructionCapability resolves an autoencoder anywhere and runs it only
// where its device runtime is built.
func reconstructionCapability() Capability {
	return inventoryCapability(autoencoderInventory, func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
		return nil, errors.New("media capability: an autoencoder's round trip runs on a CUDA device, which this build does not drive")
	})
}
