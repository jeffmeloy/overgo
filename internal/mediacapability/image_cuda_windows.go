//go:build windows

package mediacapability

import (
	"context"
	"encoding/json"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/checked"
	"overgo/internal/diffusionimage"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
	"overgo/internal/sensenovarecipe"
	"overgo/internal/strictjson"
)

const (
	imageDevice          = "cuda:0"
	imageSessionCapacity = 1
)

// pictureDirector directs sessions of a latent picture model.
type pictureDirector = capabilityruntime.ModelSessionDirector[latentimage.Request, *latentimage.Generator, latentimage.EncodedImage]

// pictureSessions makes a director of latent picture sessions. A session is
// compiled for a picture's extent, and serves any prompt and seed
// (latentimage Request.SessionKey).
func pictureSessions() (*pictureDirector, error) {
	return capabilityruntime.NewModelSessionDirector[latentimage.Request, *latentimage.Generator, latentimage.EncodedImage](
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
}

func imageCapability() Capability {
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
	latent := preparedFirst(sessionExecutor(pictureSessions()))
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
	return Capability{
		Resolve:         resolveImageSource,
		ValidationInput: imageValidationInput,
		Execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleRoutedImagePrepare:     routed,
			modelrecipe.ModuleLatentImagePrepare:     latent,
			modelrecipe.ModuleOscillatorImagePrepare: oscillator,
			modelrecipe.ModuleDiffusionImagePrepare:  diffusion,
			modelrecipe.ModuleRAEImagePrepare:        raeImage,
		}.Execute,
	}
}

// preparedFirst prepares a latent picture request ahead of its session: the pictures it names by their stored identity,
// as a page attaches them, are read into the bytes the session takes (latentimage Request.ReadPictures), and a request
// asking the MOSS Ultra workflow's prompt mode 2 has its prompt and negative written by the model's enhancer chat
// (latentimage.EnhanceRequest); it is served as the request stating what was read and written, and an input naming no
// stored picture and asking no enhancing goes to the session as it is. A request is read as strictly as the session
// reads it, so one the session would refuse (an unknown or repeated member) reaches the session as it came and is
// refused there, nothing read and no chat opened for it (SQA-408).
func preparedFirst(next capabilityruntime.Executor) capabilityruntime.Executor {
	return func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		var request latentimage.Request
		if err := strictjson.DecodeBytes([]byte(input), &request); err != nil {
			return next(ctx, store, path, selection, input)
		}
		prepared, err := request.ReadPictures(ctx, store)
		if err != nil {
			return nil, err
		}
		if request.Enhance {
			if prepared, err = latentimage.EnhanceRequest(ctx, store, path, prepared); err != nil {
				return nil, err
			}
		} else if prepared.Image == request.Image && prepared.Control == request.Control {
			return next(ctx, store, path, selection, input)
		}
		written, err := json.Marshal(prepared)
		if err != nil {
			return nil, err
		}
		return next(ctx, store, path, selection, string(written))
	}
}

func imageValidationInput(directory, prompt string) (string, error) {
	profile, err := latentimage.ResolveProfile(directory)
	if err != nil {
		return "", err
	}
	var scale int
	if profile.Rotary != nil {
		scale = profile.Rotary.SpatialScale
	} else {
		spec, err := latentimage.Derive(directory)
		if err != nil {
			return "", err
		}
		var ok bool
		scale, ok = checked.MulInt(spec.VAE.SpatialScale, spec.PatchSize)
		if !ok {
			return "", fmt.Errorf("image validation: source geometry overflows")
		}
	}
	// One stride and one step are the smallest positive source-valid image request.
	request := latentimage.Request{Prompt: prompt, Width: scale, Height: scale, Steps: 1}
	if err := latentimage.ValidateRequest(request); err != nil {
		return "", err
	}
	b, err := json.Marshal(request)
	return string(b), err
}
