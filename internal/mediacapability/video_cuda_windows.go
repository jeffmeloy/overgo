//go:build windows

package mediacapability

import (
	"context"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
)

func videoCapability() Capability {
	// Both read the base model's text encoder and tokenizer, which no
	// inventory holds, so their answers are kept under those files too.
	wan := capabilityruntime.Reading(generationExecutor(sessionExecutor(wanSessions()), sessionExecutor(studentSessions())), latentvideo.TextPipelineIdentity)

	edit := capabilityruntime.Reading(editExecutor(editSessions()), latentvideo.TextPipelineIdentity)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.VideoRequest, *oscillatorimage.Model, oscillatorimage.EncodedVideo](
		"video-gen", oscillatorimage.ValidateVideoRequest,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ oscillatorimage.VideoRequest) (*oscillatorimage.Model, error) {
			return oscillatorimage.Load(path)
		}, oscillatorimage.RegisterRuntime,
	)

	return Capability{
		Resolve: resolveVideoSource,
		Execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleLatentVideoPrepare:     wan,
			modelrecipe.ModuleReferenceVideoPrepare:  edit,
			modelrecipe.ModuleOscillatorVideoPrepare: oscillator,
		}.Execute,
	}
}

// wanDirector directs sessions of a Wan model.
type wanDirector = capabilityruntime.ModelSessionDirector[latentvideo.WanRequest, *latentvideo.WanRuntime, latentvideo.EncodedVideo]

// wanSessions makes a director of Wan sessions. A session is reset for each
// request it serves.
func wanSessions() (*wanDirector, error) {
	return capabilityruntime.NewModelSessionDirector[latentvideo.WanRequest, *latentvideo.WanRuntime, latentvideo.EncodedVideo](
		"video-gen", imageDevice, imageSessionCapacity,
		latentvideo.ValidateWanRequestForm,
		func(ctx context.Context, store artifact.Repository, path string, program recipe.Program, request latentvideo.WanRequest) (*latentvideo.WanRuntime, error) {
			return latentvideo.LoadWanRuntime(ctx, store, path, program, request)
		},
		func(ctx context.Context, runtime *latentvideo.WanRuntime, request latentvideo.WanRequest) error {
			return runtime.Reset(ctx, request)
		},
		latentvideo.RegisterWanRuntime,
	)
}

// generationExecutor serves a text-to-video request by the model it names: a few-step student of Wan through its own
// sessions, any other Wan model through the base's.
func generationExecutor(wan, student capabilityruntime.Executor) capabilityruntime.Executor {
	return func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		if latentvideo.IsWanStudent(path) {
			return student(ctx, store, path, selection, input)
		}
		return wan(ctx, store, path, selection, input)
	}
}

// studentDirector directs sessions of a few-step student of Wan.
type studentDirector = capabilityruntime.ModelSessionDirector[latentvideo.WanRequest, *latentvideo.WanStudentRuntime, latentvideo.EncodedVideo]

// studentSessions makes a director of student sessions. A session is compiled for a clip's geometry and serves any
// prompt and seed; a request of another geometry is refused at its reset, and another session loads for it.
func studentSessions() (*studentDirector, error) {
	return capabilityruntime.NewModelSessionDirector[latentvideo.WanRequest, *latentvideo.WanStudentRuntime, latentvideo.EncodedVideo](
		"video-gen", imageDevice, imageSessionCapacity,
		latentvideo.ValidateWanRequestForm,
		latentvideo.LoadWanStudentRuntime,
		func(ctx context.Context, runtime *latentvideo.WanStudentRuntime, request latentvideo.WanRequest) error {
			return runtime.Reset(ctx, request)
		},
		latentvideo.RegisterWanStudentRuntime,
	)
}

// editDirector directs sessions of a LiveEdit model.
type editDirector = capabilityruntime.ModelSessionDirector[latentvideo.ReferenceEditRequest, *latentvideo.LiveEditRuntime, latentvideo.EncodedVideo]

// editSessions makes a director of LiveEdit sessions. A session is compiled
// for a clip's shape and an edit's schedule, and serves any instruction and
// seed (latentvideo ReferenceEditRequest.SessionKey).
func editSessions() (*editDirector, error) {
	return capabilityruntime.NewMappedModelSessionDirector[latentvideo.ReferenceEditRequest, *latentvideo.LiveEditRuntime, latentvideo.EncodedVideo](
		"video-gen", imageDevice, imageSessionCapacity,
		latentvideo.ValidateReferenceEditRequest,
		func(ctx context.Context, store artifact.Repository, path string, program recipe.Program, request latentvideo.ReferenceEditRequest) (*latentvideo.LiveEditRuntime, error) {
			return latentvideo.LoadLiveEditRuntime(ctx, store, path, program, request)
		},
		func(ctx context.Context, runtime *latentvideo.LiveEditRuntime, request latentvideo.ReferenceEditRequest) error {
			return runtime.Reset(ctx, request)
		},
		latentvideo.RegisterLiveEditRuntime,
		func(request latentvideo.ReferenceEditRequest) (map[recipe.PortName]capabilityruntime.MappedInput, error) {
			// The conditioning's text context and the source clip's decoded
			// pixels are bulk: each is recorded as its settings and the digest
			// of its tensors' bytes, so a clip of any size the device holds is
			// one the workflow records, and records without writing it out.
			condition, err := capabilityruntime.InputContent("overgo.liveedit-conditioning-record.v1", request.Condition.Record())
			if err != nil {
				return nil, err
			}
			source, err := capabilityruntime.InputContent("overgo.source-video-record.v1", request.Source.Record())
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]capabilityruntime.MappedInput{
				"condition": {Value: request.Condition, Content: condition},
				"source":    {Value: request.Source, Content: source},
			}, nil
		},
	)
}

// editExecutor serves LiveEdit requests through a director: a clip-form
// request resolved and handed over as the value it is, a resolved one as the
// text it arrived as.
func editExecutor(director *editDirector, err error) capabilityruntime.Executor {
	return resolveEditForm(latentvideo.ResolveClipRequest, sessionExecutor(director, err),
		func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, request latentvideo.ReferenceEditRequest, kept artifact.Content) (any, error) {
			if err != nil {
				return nil, err
			}
			return director.ExecuteInput(ctx, store, path, selection, request, kept)
		})
}

func resolveVideoSource(ctx context.Context, path string) (Source, error) {
	generation := func(modelID, profileID artifact.ID) (recipe.Definition, error) {
		return modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, modelID, profileID)
	}
	if latentvideo.IsWanStudent(path) {
		// A student is the base's DiT distilled: it reads its own weights and the base's configuration and
		// autoencoder, and generates from words alone, as the base does.
		wan, checkpoint, err := latentvideo.WanStudentArtifacts(path)
		if err != nil {
			return Source{}, err
		}
		return compositeSource(path, sharedDirectory(path, wan), []modelartifact.FileSpec{
			{Path: checkpoint, Name: "student/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "config.json"), Name: "wan/config", Role: artifact.ComponentConfig},
			{Path: latentvideo.AutoencoderWeights(wan), Name: "wan/vae", Role: artifact.ComponentWeights},
		}, profiledDefinition(path, generation))
	}
	if latentvideo.IsLiveEdit(path) {
		wan := filepath.Join(filepath.Dir(path), "Wan2.1-T2V-1.3B")
		// The edit composite genuinely spans two sibling repositories --
		// the LiveEdit weights and the Wan denoiser they condition on --
		// so the inventory roots at their common parent.
		return compositeSource(path, filepath.Dir(path), []modelartifact.FileSpec{
			{Path: filepath.Join(path, "ar-forcing_002000.pt"), Name: "liveedit/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "config.json"), Name: "wan/config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(wan, "diffusion_pytorch_model.safetensors"), Name: "wan/weights", Role: artifact.ComponentWeights},
			{Path: latentvideo.AutoencoderWeights(wan), Name: "wan/vae", Role: artifact.ComponentWeights},
		}, profiledDefinition(wan, modelrecipe.ReferenceVideoEditDefinition))
	}
	if latentvideo.IsWan(path) {
		inventory, err := modelartifact.FromFiles(path, []modelartifact.FileSpec{
			{Path: filepath.Join(path, "config.json"), Name: "config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(path, "diffusion_pytorch_model.safetensors"), Name: "denoiser/weights", Role: artifact.ComponentWeights},
			{Path: latentvideo.AutoencoderWeights(path), Name: "vae/weights", Role: artifact.ComponentWeights},
		})
		return Source{Inventory: inventory, Define: profiledDefinition(path, generation)}, err
	}
	return resolveDeclaredVideoSource(ctx, path)
}

// compositeSource inventories a model whose files span its own directory and its base's, rooted at root, which holds
// both. The capability resolves only from the model's own directory, so that is recorded beside the root; the
// identity is the components' alone.
func compositeSource(path, root string, specs []modelartifact.FileSpec, define func(artifact.ID) (recipe.Definition, []artifact.Content, error)) (Source, error) {
	inventory, err := modelartifact.FromFiles(root, specs)
	if err == nil {
		var own artifact.Location
		own, err = artifact.CanonicalLocalLocation(inventory.Manifest.ID, artifact.LocationDirectory, path)
		inventory.Locations = append(inventory.Locations, own)
	}
	return Source{Inventory: inventory, Define: define}, err
}

// sharedDirectory is the deepest directory holding both paths: a student and its base may sit in different models
// directories.
func sharedDirectory(left, right string) string {
	directory := filepath.Dir(left)
	for directory != filepath.Dir(directory) {
		if relative, err := filepath.Rel(directory, right); err == nil && filepath.IsLocal(relative) {
			break
		}
		directory = filepath.Dir(directory)
	}
	return directory
}

// profiledDefinition defines a model by the video profile the directory resolves to, published beside it.
func profiledDefinition(directory string, define func(modelID, profileID artifact.ID) (recipe.Definition, error)) func(artifact.ID) (recipe.Definition, []artifact.Content, error) {
	return func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
		profile, err := latentvideo.ResolveProfile(directory)
		if err != nil {
			return recipe.Definition{}, nil, err
		}
		content, err := profile.Content()
		if err != nil {
			return recipe.Definition{}, nil, err
		}
		definition, err := define(modelID, profile.ID)
		return definition, []artifact.Content{content}, err
	}
}

// deviceReconstruction keeps a Wan autoencoder resident on the device and
// passes each request's picture through it.
func deviceReconstruction() capabilityruntime.Executor {
	director, err := capabilityruntime.NewModelSessionDirector[latentvideo.ReconstructionRequest, *latentvideo.AutoencoderRuntime, latentimage.EncodedImage](
		"reconstruction", imageDevice, imageSessionCapacity,
		latentvideo.ValidateReconstructionRequest,
		capabilityruntime.IgnoreInput[latentvideo.ReconstructionRequest](latentvideo.LoadAutoencoderRuntime),
		func(ctx context.Context, runtime *latentvideo.AutoencoderRuntime, request latentvideo.ReconstructionRequest) error {
			return runtime.Reset(ctx, request)
		},
		latentvideo.RegisterAutoencoderRuntime,
	)
	return sessionExecutor(director, err)
}

// openDeviceAutoencoder loads the Wan autoencoder in a directory onto the
// device and holds it until the caller closes it.
func openDeviceAutoencoder(path string) (Autoencoder, error) {
	runtime, err := latentvideo.LoadAutoencoderRuntime(path)
	if err != nil {
		return nil, err
	}
	return runtime, nil
}
