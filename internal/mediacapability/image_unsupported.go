//go:build !windows

package mediacapability

import (
	"overgo/internal/capabilityruntime"
	"overgo/internal/diffusionimage"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
)

func imageCapability() Capability {
	diffusion := capabilityruntime.JSONScalar[diffusionimage.Request, *diffusionimage.Model, latentimage.EncodedImage](
		"image-gen", diffusionimage.ValidateRequest,
		capabilityruntime.IgnoreInput[diffusionimage.Request](diffusionimage.Load), diffusionimage.RegisterRuntime[*diffusionimage.Model],
	)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, latentimage.EncodedImage](
		"image-gen", oscillatorimage.ValidateRequest,
		capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
	)
	return Capability{
		Resolve: resolveImageSource,
		Execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleDiffusionImagePrepare:  diffusion,
			modelrecipe.ModuleOscillatorImagePrepare: oscillator,
		}.Execute,
	}
}
