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
	"overgo/internal/enhancerchat"
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
	latent := capabilityruntime.Reading(preparedFirst(sessionExecutor(pictureSessions())), capabilityruntime.Contents(pictureFiles))
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
		Preview: capabilityruntime.ExecutorCatalog{modelrecipe.ModuleLatentImagePrepare: depthPreview}.Execute,
	}
}

// depthPreview is the depth map a latent picture request's depth control samples by, as the MOSS Ultra workflow
// previews it (latentimage.DepthPreview), the pictures the request names by their stored identity read as its run
// reads them.
func depthPreview(ctx context.Context, store artifact.Repository, path string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
	var request latentimage.Request
	if err := strictjson.DecodeBytes([]byte(input), &request); err != nil {
		return nil, err
	}
	read, err := request.ReadPictures(ctx, store)
	if err != nil {
		return nil, err
	}
	return latentimage.DepthPreview(path, read)
}

// enhancerOpener opens a MOSS model's enhancer chat through the store's profiles (enhancerchat.Open), sampled and
// trimmed as the picture package asks (latentimage.EnhancerOpener).
func enhancerOpener(store artifact.Repository) latentimage.EnhancerOpener {
	return func(ctx context.Context, directory string) (func(context.Context, string, string, int) (string, error), func() error, error) {
		return enhancerchat.Open(ctx, store, directory, latentimage.EnhancerSampling(), latentimage.PythonSpace)
	}
}

// pictureFiles are the files a latent picture request reads beside its model's components, which its answer is kept
// under as well (capabilityruntime.Reading): the adapters it or its preset names and the depth estimator a control
// picture is read by (latentimage Request.Beside). A file replaced under its name gave the earlier file's picture again
// (SQA-406). A request that cannot be read names none, and the session refuses it.
func pictureFiles(path, raw string) ([]string, error) {
	var request latentimage.Request
	if strictjson.DecodeBytes([]byte(raw), &request) != nil {
		return nil, nil
	}
	if _, err := admitted(path, request); err != nil {
		return nil, nil
	}
	return request.Beside(path)
}

// admitted is a latent picture request sized as its session takes it (latentimage Request.Sized), or the refusal its
// session would give it (latentimage.Admit), before any file it names is read (SQA-425).
func admitted(path string, request latentimage.Request) (latentimage.Request, error) {
	sized, err := request.Sized(path)
	if err != nil {
		return latentimage.Request{}, err
	}
	return sized, latentimage.Admit(sized)
}

// preparedFirst prepares a latent picture request ahead of its session: a request stating no size is given its
// canvas's or its preset's (latentimage Request.Sized), since the session is chosen and checked by its size before its
// model loads (SQA-409); the pictures it names by their stored identity, as a page attaches them, are read into the
// bytes the session takes (Request.ReadPictures); and a request asking the MOSS Ultra workflow's prompt mode 2 has its
// prompt and negative written by the model's enhancer chat (latentimage.EnhanceRequest). It is served as the request
// stating what was given, read and written, and an input needing none of them goes to the session as it is. A request is
// read as strictly as the session reads it, so one the session would refuse (an unknown or repeated member) reaches the
// session as it came and is refused there, nothing read and no chat opened for it (SQA-408); one it reads that the
// session would refuse once sized is refused here as the session would refuse it, nothing read (admitted).
func preparedFirst(next capabilityruntime.Executor) capabilityruntime.Executor {
	return func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		var request latentimage.Request
		if err := strictjson.DecodeBytes([]byte(input), &request); err != nil {
			return next(ctx, store, path, selection, input)
		}
		sized, err := admitted(path, request)
		if err != nil {
			return nil, err
		}
		prepared, err := sized.ReadPictures(ctx, store)
		if err != nil {
			return nil, err
		}
		if request.Enhance {
			if prepared, err = latentimage.EnhanceRequest(ctx, enhancerOpener(store), path, prepared); err != nil {
				return nil, err
			}
		} else if prepared.Image == request.Image && prepared.Control == request.Control && prepared.Width == request.Width && prepared.Canvas == request.Canvas {
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
