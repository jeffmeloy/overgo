//go:build windows

package main

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/diffusionimage"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
	"overgo/internal/sensenovarecipe"
)

const (
	imageDevice          = "cuda:0"
	imageSessionCapacity = 1
)

func imageCapability() capability {
	routedDirector, routedErr := capabilityruntime.NewModelSessionDirector[sensenovarecipe.GenerationRequest, *sensenovarecipe.Generator, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity,
		sensenovarecipe.ValidateGenerationRequest,
		func(ctx context.Context, store artifact.Repository, path string, program recipe.Program, _ sensenovarecipe.GenerationRequest) (*sensenovarecipe.Generator, error) {
			return sensenovarecipe.LoadGenerator(ctx, store, path, program.Definition())
		},
		func(ctx context.Context, generator *sensenovarecipe.Generator, request sensenovarecipe.GenerationRequest) error {
			return generator.Reset(ctx, request)
		},
		sensenovarecipe.RegisterRuntime,
	)
	routed := sessionExecutor(routedDirector, routedErr)
	latentDirector, err := capabilityruntime.NewModelSessionDirector[latentimage.Request, *latentimage.Generator, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity,
		latentimage.ValidateRequest,
		func(ctx context.Context, store artifact.Repository, path string, program recipe.Program, request latentimage.Request) (*latentimage.Generator, error) {
			profileID, ok := program.Definition().PrimaryDependency(recipe.DependencyProfile)
			if !ok {
				return nil, fmt.Errorf("image-gen: compiled recipe has no profile")
			}
			profile, err := latentimage.ReadProfile(ctx, store, profileID)
			if err != nil {
				return nil, err
			}
			return latentimage.LoadGenerator(ctx, path, profile, request)
		},
		func(ctx context.Context, generator *latentimage.Generator, request latentimage.Request) error {
			return generator.Reset(ctx, request)
		},
		latentimage.RegisterRuntime,
	)
	latent := sessionExecutor(latentDirector, err)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, latentimage.EncodedImage](
		"image-gen", oscillatorimage.ValidateRequest,
		capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
	)
	diffusionDirector, diffusionErr := capabilityruntime.NewModelSessionDirector[diffusionimage.Request, *diffusionimage.ResidentGenerator, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity,
		diffusionimage.ValidateRequest,
		func(ctx context.Context, _ artifact.Repository, path string, _ recipe.Program, request diffusionimage.Request) (*diffusionimage.ResidentGenerator, error) {
			return diffusionimage.LoadResidentGenerator(ctx, path, request)
		},
		func(ctx context.Context, generator *diffusionimage.ResidentGenerator, request diffusionimage.Request) error {
			return generator.Reset(ctx, request)
		},
		diffusionimage.RegisterRuntime[*diffusionimage.ResidentGenerator],
	)
	diffusion := sessionExecutor(diffusionDirector, diffusionErr)
	return capability{
		resolve: resolveImageSource,
		execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleRoutedImagePrepare:     routed,
			modelrecipe.ModuleLatentImagePrepare:     latent,
			modelrecipe.ModuleOscillatorImagePrepare: oscillator,
			modelrecipe.ModuleDiffusionImagePrepare:  diffusion,
		}.Execute,
	}
}
