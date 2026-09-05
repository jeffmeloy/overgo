//go:build windows

package mediacapability

import (
	"context"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
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
			contents, err := latentvideo.ReferenceEditInputContent(request)
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]capabilityruntime.MappedInput{
				"condition": {Value: request.Condition, Content: contents["condition"]},
				"source":    {Value: request.Source, Content: contents["source"]},
			}, nil
		},
	)
	edit := sessionExecutor(editDirector, editErr)
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
		// so the inventory roots at their common parent.
		inventory, err := modelartifact.FromFiles(filepath.Dir(path), []modelartifact.FileSpec{
			{Path: filepath.Join(path, "ar-forcing_002000.pt"), Name: "liveedit/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "config.json"), Name: "wan/config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(wan, "diffusion_pytorch_model.safetensors"), Name: "wan/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "Wan2.1_VAE.pth"), Name: "wan/vae", Role: artifact.ComponentWeights},
		})
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
			{Path: filepath.Join(path, "Wan2.1_VAE.pth"), Name: "vae/weights", Role: artifact.ComponentWeights},
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
