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
	"overgo/internal/heldfile"
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
// (latentimage Request.SessionKey). It takes a request as its session will
// take it (latentimage.Admit): one asking prompt mode 2 as it will be once its
// enhancer has written it, which it does only where no answer is stored
// (pictureExecutor).
func pictureSessions() (*pictureDirector, error) {
	return capabilityruntime.NewModelSessionDirector[latentimage.Request, *latentimage.Generator, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity,
		latentimage.Admit,
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

// pictureExecutor serves latent picture requests through director (err its construction's): each sized by its
// selected profile's preset and admitted before anything it names is read (preparedFirst), its answer kept under the
// files it reads beside its model, named through held (pictureFiles), and sought in the store by the request as it
// names its pictures, by their stored identity, which are read only where no answer is stored (readPictures,
// ReadsOnMiss): a repeated picture had read and encoded its source picture before its answer was sought
// (moss-served-repeat); so is prompt mode 2's enhancement, its chat opened through opened (enhancerOpener where it
// serves), the request sought by its idea and its enhancer's files (onMiss, enhanced-repeat, SQA-465). A sharpened
// picture comes with the picture it drew (keptDrawn). Model is the session's model, a latentimage Generator where it
// serves.
func pictureExecutor[Model any](director *capabilityruntime.ModelSessionDirector[latentimage.Request, Model, latentimage.EncodedImage], err error, held *heldfile.Set, opened func(artifact.Repository) latentimage.EnhancerOpener) capabilityruntime.Executor {
	if err == nil {
		director.ReadsOnMiss(onMiss(opened))
	}
	return keptDrawn(capabilityruntime.Reading(preparedFirst(sessionExecutor(director, err)), capabilityruntime.Contents(held, pictureFiles)))
}

// keptDrawn has each sharpened picture next serves come with the picture as it was drawn, before its finishing
// sharpened it, as the MOSS Ultra workflow saves both and compares them (moss-sharpened-pair): a picture drawn holds it
// (latentimage EncodedImage.Drawn), which is stored beside the sharpened picture as the one it was derived from, and a
// picture its stored run answers, which holds the sharpened picture alone, is given the picture it was derived from.
func keptDrawn(next capabilityruntime.Executor) capabilityruntime.Executor {
	return func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		output, err := next(ctx, store, path, selection, input)
		measured, enveloped := output.(capabilityruntime.Measured)
		if err != nil || !enveloped || store == nil {
			return output, err
		}
		picture, drawn := measured.Output.(latentimage.EncodedImage)
		if !drawn {
			return output, nil
		}
		finished, err := latentimage.PNGContent(picture)
		if err != nil {
			return nil, err
		}
		// The picture is returned as its bytes are stored, what it was drawn from written into them (PNGContent), so a
		// picture drawn and the same picture its stored run answers are one.
		picture.Data = finished.Data
		measured.Output = picture
		output = measured
		if before, sharpened := picture.BeforeFinishing(); sharpened {
			first, err := latentimage.PNGContent(before.(latentimage.EncodedImage))
			if err != nil {
				return nil, err
			}
			if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: "picture-drawn/" + finished.Descriptor.ID.String(),
				Artifacts: []artifact.Descriptor{first.Descriptor, finished.Descriptor}, Contents: []artifact.Content{first, finished},
				Lineage: []artifact.Lineage{{Child: finished.Descriptor.ID, Parent: first.Descriptor.ID, Relation: artifact.RelationDerivedFrom}}}); err != nil {
				return nil, err
			}
			return output, nil
		}
		// A picture its stored run answers says it was sharpened by its own bytes (latentimage EncodedImage.Sharpened); its
		// request, which a request above a mebibyte is stored as the digest of, is never read back (SQA-464).
		if !picture.Sharpened {
			return output, nil
		}
		parents, err := store.Parents(ctx, finished.Descriptor.ID)
		if err != nil {
			return nil, err
		}
		for _, edge := range parents {
			if edge.Relation != artifact.RelationDerivedFrom {
				continue
			}
			content, held, err := artifact.ReadContent(ctx, store, edge.Parent)
			if err != nil {
				return nil, err
			}
			if held && content.Descriptor.MediaType == picture.MediaType {
				picture.Drawn = content.Data
				measured.Output = picture
				return measured, nil
			}
		}
		return output, nil
	}
}

// onMiss is what a picture request's director does where it holds no answer to the request: the pictures it names by
// their stored identity read into the bytes its session takes (latentimage Request.ReadPictures), and, for prompt mode
// 2, its prompt and negative written by the model's enhancer chat, opened through opened (latentimage.EnhanceRequest),
// so a repeat answered by its stored run opens no chat (enhanced-repeat, SQA-465).
func onMiss(opened func(artifact.Repository) latentimage.EnhancerOpener) func(context.Context, artifact.Repository, string, latentimage.Request) (latentimage.Request, error) {
	return func(ctx context.Context, store artifact.Repository, path string, request latentimage.Request) (latentimage.Request, error) {
		read, err := request.ReadPictures(ctx, store)
		if err != nil {
			return latentimage.Request{}, err
		}
		return latentimage.EnhanceRequest(ctx, opened(store), path, read)
	}
}

// pictureCatalog is the catalog a picture request's preset is read from: the selected profile's, which the store holds
// beside its recipe (latentimage.CatalogOf), so serving a request reads no model_index.json (moss-served-repeat); or,
// for a selection naming no profile, the model directory's (latentimage.CatalogIn).
func pictureCatalog(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection) (latentimage.Catalog, error) {
	profileID, named := selection.Program.Definition().PrimaryDependency(recipe.DependencyProfile)
	if !named || store == nil {
		return latentimage.CatalogIn(path), nil
	}
	profile, err := latentimage.ReadProfile(ctx, store, profileID)
	if err != nil {
		return latentimage.Catalog{}, err
	}
	return latentimage.CatalogOf(profile), nil
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
	pictures, picturesErr := pictureSessions()
	latent := pictureExecutor(pictures, picturesErr, heldfile.Session, enhancerOpener)
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
// (SQA-406). Its preset is read from the selected profile (pictureCatalog). A request that cannot be read names none,
// and the session refuses it.
func pictureFiles(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, raw string) ([]string, error) {
	var request latentimage.Request
	if strictjson.DecodeBytes([]byte(raw), &request) != nil {
		return nil, nil
	}
	catalog, err := pictureCatalog(ctx, store, path, selection)
	if err != nil {
		return nil, nil
	}
	if _, err := admitted(catalog, request); err != nil {
		return nil, nil
	}
	return request.Beside(path, catalog)
}

// admitted is a latent picture request sized as its session takes it, its preset read from catalog (latentimage
// Request.Sized), or the refusal its session would give it (latentimage.Admit), before any file it names is read
// (SQA-425).
func admitted(catalog latentimage.Catalog, request latentimage.Request) (latentimage.Request, error) {
	sized, err := request.Sized(catalog)
	if err != nil {
		return latentimage.Request{}, err
	}
	return sized, latentimage.Admit(sized)
}

// preparedFirst prepares a latent picture request ahead of its session: a request stating no size is given its
// canvas's or its preset's, the preset read from the selected profile (latentimage Request.Sized, pictureCatalog),
// since the session is chosen and checked by its size before its model loads (SQA-409). It is served as the request
// stating the size it was given, and an input needing none goes to the session as it is. The pictures it names by their
// stored identity, as a page attaches them, and a prompt mode 2 idea go on so named: the session's director seeks its
// stored answer by them and reads the pictures, and has the enhancer write the prompt, only where it holds none
// (pictureExecutor, onMiss). A request is read as strictly as the session reads it, so one the session would refuse (an
// unknown or repeated member) reaches the session as it came and is refused there, no chat opened for it (SQA-408); one
// it reads that the session would refuse once sized is refused here as the session would refuse it (admitted).
func preparedFirst(next capabilityruntime.Executor) capabilityruntime.Executor {
	return func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		var request latentimage.Request
		if err := strictjson.DecodeBytes([]byte(input), &request); err != nil {
			return next(ctx, store, path, selection, input)
		}
		catalog, err := pictureCatalog(ctx, store, path, selection)
		if err != nil {
			return nil, err
		}
		prepared, err := admitted(catalog, request)
		if err != nil {
			return nil, err
		}
		if prepared.Width == request.Width && prepared.Canvas == request.Canvas {
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
