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

// errNoAutoencoderDevice is why an autoencoder resolves here and does not run.
var errNoAutoencoderDevice = errors.New("media capability: an autoencoder's round trip runs on a CUDA device, which this build does not drive")

// deviceReconstruction refuses: a Wan autoencoder resolves anywhere and runs
// only where its device runtime is built.
func deviceReconstruction() capabilityruntime.Executor {
	return func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
		return nil, errNoAutoencoderDevice
	}
}

// openDeviceAutoencoder refuses: the device runtime is not built here.
func openDeviceAutoencoder(string) (Autoencoder, error) { return nil, errNoAutoencoderDevice }
