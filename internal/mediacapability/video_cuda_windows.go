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
	wanDirector, wanErr := capabilityruntime.NewModelSessionDirector[latentvideo.WanRequest, *latentvideo.WanRuntime, latentvideo.EncodedVideo](
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
	wan := sessionExecutor(wanDirector, wanErr)

	editDirector, editErr := capabilityruntime.NewMappedModelSessionDirector[latentvideo.ReferenceEditRequest, *latentvideo.LiveEditRuntime, latentvideo.EncodedVideo](
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
			// pixels are bulk: each is recorded by its digest once it is
			// over the inline bound, so a clip of any size the device holds
			// is one the workflow can record.
			condition, err := capabilityruntime.InputContent("overgo.liveedit-conditioning.v1", request.Condition)
			if err != nil {
				return nil, err
			}
			source, err := capabilityruntime.InputContent("overgo.source-video.v1", request.Source)
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]capabilityruntime.MappedInput{
				"condition": {Value: request.Condition, Content: condition},
				"source":    {Value: request.Source, Content: source},
			}, nil
		},
	)
	edit := resolveEditForm(latentvideo.ResolveClipRequest, sessionExecutor(editDirector, editErr))
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

func resolveVideoSource(path string) (Source, error) {
	if latentvideo.IsLiveEdit(path) {
		wan := filepath.Join(filepath.Dir(path), "Wan2.1-T2V-1.3B")
		// The edit composite genuinely spans two sibling repositories --
		// the LiveEdit weights and the Wan denoiser they condition on --
		// so the inventory roots at their common parent. The capability
		// resolves only from the LiveEdit directory, so that is recorded
		// beside the root; the identity is the components' alone.
		inventory, err := modelartifact.FromFiles(filepath.Dir(path), []modelartifact.FileSpec{
			{Path: filepath.Join(path, "ar-forcing_002000.pt"), Name: "liveedit/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "config.json"), Name: "wan/config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(wan, "diffusion_pytorch_model.safetensors"), Name: "wan/weights", Role: artifact.ComponentWeights},
			{Path: latentvideo.AutoencoderWeights(wan), Name: "wan/vae", Role: artifact.ComponentWeights},
		})
		if err == nil {
			var own artifact.Location
			own, err = artifact.CanonicalLocalLocation(inventory.Manifest.ID, artifact.LocationDirectory, path)
			inventory.Locations = append(inventory.Locations, own)
		}
		return Source{Inventory: inventory, Define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
			profile, err := latentvideo.ResolveProfile(wan)
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			content, err := profile.Content()
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			definition, err := modelrecipe.ReferenceVideoEditDefinition(modelID, profile.ID)
			return definition, []artifact.Content{content}, err
		}}, err
	}
	if latentvideo.IsWan(path) {
		inventory, err := modelartifact.FromFiles(path, []modelartifact.FileSpec{
			{Path: filepath.Join(path, "config.json"), Name: "config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(path, "diffusion_pytorch_model.safetensors"), Name: "denoiser/weights", Role: artifact.ComponentWeights},
			{Path: latentvideo.AutoencoderWeights(path), Name: "vae/weights", Role: artifact.ComponentWeights},
		})
		return Source{Inventory: inventory, Define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
			profile, err := latentvideo.ResolveProfile(path)
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			content, err := profile.Content()
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, modelID, profile.ID)
			return definition, []artifact.Content{content}, err
		}}, err
	}
	return resolveDeclaredVideoSource(path)
}

// reconstructionCapability keeps the autoencoder resident on the device and
// passes each request's picture through it.
func reconstructionCapability() Capability {
	director, err := capabilityruntime.NewModelSessionDirector[latentvideo.ReconstructionRequest, *latentvideo.AutoencoderRuntime, latentimage.EncodedImage](
		"reconstruction", imageDevice, imageSessionCapacity,
		latentvideo.ValidateReconstructionRequest,
		capabilityruntime.IgnoreInput[latentvideo.ReconstructionRequest](latentvideo.LoadAutoencoderRuntime),
		func(ctx context.Context, runtime *latentvideo.AutoencoderRuntime, request latentvideo.ReconstructionRequest) error {
			return runtime.Reset(ctx, request)
		},
		latentvideo.RegisterAutoencoderRuntime,
	)
	return inventoryCapability(autoencoderInventory, sessionExecutor(director, err))
}

// OpenAutoencoder loads the autoencoder in a directory onto the device and
// holds it until the caller closes it.
func OpenAutoencoder(path string) (Autoencoder, error) {
	runtime, err := latentvideo.LoadAutoencoderRuntime(path)
	if err != nil {
		return nil, err
	}
	return runtime, nil
}
