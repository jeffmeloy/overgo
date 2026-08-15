//go:build windows

package main

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentvideo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
)

func videoCapability() capability {
	wanCache, wanErr := capabilityruntime.NewScalarSessionCache[latentvideo.WanRequest, *latentvideo.WanRuntime, latentvideo.EncodedVideo](
		"video-gen", imageDevice, imageSessionCapacity,
		latentvideo.ValidateWanRequest, latentvideo.WanSessionPolicy,
		func(ctx context.Context, store artifact.Repository, path string, program recipe.Program, request latentvideo.WanRequest) (*latentvideo.WanRuntime, error) {
			return latentvideo.LoadWanRuntime(ctx, store, path, program, request)
		},
		func(ctx context.Context, runtime *latentvideo.WanRuntime, request latentvideo.WanRequest) error {
			return runtime.Reset(ctx, request)
		},
		latentvideo.RegisterWanRuntime,
	)
	wan := residentExecutor(wanCache, wanErr)

	editCache, editErr := capabilityruntime.NewMappedSessionCache[latentvideo.ReferenceEditRequest, *latentvideo.LiveEditRuntime, latentvideo.EncodedVideo](
		"video-gen", imageDevice, imageSessionCapacity,
		latentvideo.ValidateReferenceEditRequest, latentvideo.ReferenceEditSessionPolicy,
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
	edit := residentExecutor(editCache, editErr)
	oscillator := capabilityruntime.JSONScalar[oscillatorimage.VideoRequest, *oscillatorimage.Model, oscillatorimage.EncodedVideo](
		"video-gen", oscillatorimage.ValidateVideoRequest,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ oscillatorimage.VideoRequest) (*oscillatorimage.Model, error) {
			return oscillatorimage.Load(path)
		}, oscillatorimage.RegisterVideoRuntime,
	)

	return capability{
		inventory: videoInventory,
		execute: func(ctx context.Context, store artifact.Repository, path string, modelID artifact.ID, program recipe.Program, raw string) (any, error) {
			switch latentvideo.VideoProgramModule(program) {
			case modelrecipe.ModuleLatentVideoPrepare:
				return wan(ctx, store, path, modelID, program, raw)
			case modelrecipe.ModuleReferenceVideoPrepare:
				return edit(ctx, store, path, modelID, program, raw)
			case modelrecipe.ModuleOscillatorVideoPrepare:
				return oscillator(ctx, store, path, modelID, program, raw)
			default:
				return nil, errors.New("video-gen: compiled recipe has no registered operator")
			}
		},
		bind: func(path string, modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
			if latentvideo.IsLiveEdit(path) {
				profile, err := latentvideo.ResolveProfile(filepath.Join(filepath.Dir(path), "Wan2.1-T2V-1.3B"))
				if err != nil {
					return recipe.Definition{}, nil, err
				}
				content, err := profile.Content()
				if err != nil {
					return recipe.Definition{}, nil, err
				}
				definition, err := modelrecipe.ReferenceVideoEditDefinition(modelID, profile.ID)
				return definition, []artifact.Content{content}, err
			}
			if latentvideo.IsWan(path) {
				profile, err := latentvideo.ResolveProfile(path)
				if err != nil {
					return recipe.Definition{}, nil, err
				}
				content, err := profile.Content()
				if err != nil {
					return recipe.Definition{}, nil, err
				}
				definition, err := modelrecipe.LatentVideoDefinition(modelID, profile.ID)
				return definition, []artifact.Content{content}, err
			}
			recognized, err := oscillatorimage.Recognize(path)
			if err != nil {
				return recipe.Definition{}, nil, err
			}
			if !recognized {
				return recipe.Definition{}, nil, errors.New("video-gen: artifact has no registered video recipe")
			}
			definition, err := modelrecipe.OscillatorVideoDefinition(modelID)
			return definition, nil, err
		},
	}
}

func videoInventory(path string) (modelartifact.Inventory, error) {
	if latentvideo.IsLiveEdit(path) {
		wan := filepath.Join(filepath.Dir(path), "Wan2.1-T2V-1.3B")
		return modelartifact.FromFiles(path, []modelartifact.FileSpec{
			{Path: filepath.Join(path, "ar-forcing_002000.pt"), Name: "liveedit/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "config.json"), Name: "wan/config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(wan, "diffusion_pytorch_model.safetensors"), Name: "wan/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "Wan2.1_VAE.pth"), Name: "wan/vae", Role: artifact.ComponentWeights},
		})
	}
	if latentvideo.IsWan(path) {
		return modelartifact.FromFiles(path, []modelartifact.FileSpec{
			{Path: filepath.Join(path, "config.json"), Name: "config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(path, "diffusion_pytorch_model.safetensors"), Name: "denoiser/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(path, "Wan2.1_VAE.pth"), Name: "vae/weights", Role: artifact.ComponentWeights},
		})
	}
	recognized, err := oscillatorimage.Recognize(path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	if !recognized {
		return modelartifact.Inventory{}, fmt.Errorf("video-gen: artifact has no registered video recipe")
	}
	return imageGenInventory(path)
}
