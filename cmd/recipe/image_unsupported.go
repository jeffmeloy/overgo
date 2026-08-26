//go:build !windows

package main

import (
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
		resolve: resolveImageSource,
		execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleDiffusionImagePrepare:  diffusion,
			modelrecipe.ModuleOscillatorImagePrepare: oscillator,
		}.Execute,
	}
}
