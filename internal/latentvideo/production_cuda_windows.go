//go:build windows

package latentvideo

import (
	"context"
	"errors"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/media"
	"overgo/internal/recipe"
	"overgo/internal/tensor/dtype"
	"overgo/internal/workflowruntime"
)

// WanRuntime retains one generator across compatible requests.
type WanRuntime struct {
	generator *Generator
	profile   Profile
	runs      int
}

func LoadWanRuntime(ctx context.Context, store artifact.Reader, path string, program recipe.Program, request WanRequest) (*WanRuntime, error) {
	profile, err := readProgramProfile(ctx, store, program)
	if err != nil {
		return nil, err
	}
	storage := dtype.BF16
	if !checked.Equal(profile.Precision.MatmulWeights, dtype.BF16.String()) {
		return nil, errors.New("latent video: unsupported production precision")
	}
	generator, err := NewGenerator(GeneratorConfig{
		ModelDirectory: path, Policy: profile.Policy, LatentStats: profile.LatentStats,
		Frames: request.Frames, Width: request.Width, Height: request.Height,
		Precision: DenoiserPrecision{
			MatmulWeights: storage, RoundAttentionStorage: profile.Precision.RoundAttentionStorage,
		}, DeviceOrdinal: device.DefaultOrdinal(),
	})
	if err != nil {
		return nil, err
	}
	return &WanRuntime{generator: generator, profile: profile}, nil
}

func (r *WanRuntime) Reset(_ context.Context, request WanRequest) error {
	if r == nil || r.generator == nil {
		return errors.New("latent video: Wan runtime is closed")
	}
	geometry := r.generator.Geometry()
	volume, err := media.DownsampledVolume(request.Frames, request.Height, request.Width, r.profile.Policy.VAEStride)
	if err != nil {
		return err
	}
	if !checked.Equal(geometry.LatentFrames, volume.Frames) {
		return errors.New("latent video: request differs from resident Wan geometry")
	}
	if !checked.Equal(geometry.LatentHeight, volume.Height) {
		return errors.New("latent video: request differs from resident Wan geometry")
	}
	if !checked.Equal(geometry.LatentWidth, volume.Width) {
		return errors.New("latent video: request differs from resident Wan geometry")
	}
	return nil
}

func (r *WanRuntime) Generate(ctx context.Context, request WanRequest) (EncodedVideo, error) {
	sink, err := NewGIFEncoder(r.profile.SampleFPS, SignedUnitPixels)
	if err != nil {
		return EncodedVideo{}, err
	}
	_, err = r.generator.Generate(ctx, GenerateRequest{
		Steps: request.Steps, Shift: request.Shift, GuideScale: request.GuideScale,
		CondContext: request.CondContext, UncondContext: request.UncondContext,
		InitialSample: request.InitialSample, Noise: request.Noise, Sink: sink.Add,
	})
	if err != nil {
		return EncodedVideo{}, err
	}
	video, err := sink.Finish()
	if err == nil {
		r.runs++
		video.ResidentRun = r.runs
	}
	return video, err
}

func (r *WanRuntime) Close(context.Context) error {
	if r == nil || r.generator == nil {
		return nil
	}
	err := r.generator.Close()
	r.generator = nil
	return err
}

func RegisterWanRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *WanRuntime) error {
	if model == nil || model.generator == nil {
		return errors.New("latent video: incomplete Wan runtime binding")
	}
	return registerWanRuntime(runtime, modelID, model.Generate)
}

// LiveEditRuntime retains source codec, denoiser history graphs, and decoder.
type LiveEditRuntime struct {
	runtime *ReferenceEditRuntime
	profile Profile
	runs    int
}

func LoadLiveEditRuntime(ctx context.Context, store artifact.Reader, path string, program recipe.Program, request ReferenceEditRequest) (*LiveEditRuntime, error) {
	profile, err := readProgramProfile(ctx, store, program)
	if err != nil {
		return nil, err
	}
	wanDirectory := filepath.Join(filepath.Dir(path), "Wan2.1-T2V-1.3B")
	base, err := LoadDenoiserConfig(wanDirectory, profile.Policy)
	if err != nil {
		return nil, err
	}
	source := request.Source
	condition := request.Condition
	runtime, err := NewReferenceEditRuntime(ReferenceEditRuntimeConfig{
		WanDirectory: wanDirectory, EditCheckpoint: filepath.Join(path, "ar-forcing_002000.pt"),
		Policy: profile.Policy, LatentStats: profile.LatentStats,
		Source:         SourceVideoShape{Channels: source.Channels, Frames: source.Frames, Height: source.Height, Width: source.Width},
		FramesPerChunk: condition.FramesPerChunk, LocalAttentionFrames: condition.LocalAttention,
		Timesteps: condition.Timesteps, Sigmas: condition.Sigmas, ContextTimestep: condition.ContextTimestep,
		Layers: base.NumLayers, DeviceOrdinal: device.DefaultOrdinal(), Seed: condition.Seed, TextContext: condition.TextContext,
	})
	if err != nil {
		return nil, err
	}
	return &LiveEditRuntime{runtime: runtime, profile: profile}, nil
}

func (r *LiveEditRuntime) Reset(context.Context, ReferenceEditRequest) error {
	if r == nil || r.runtime == nil {
		return errors.New("latent video: LiveEdit runtime is closed")
	}
	return nil
}

func (r *LiveEditRuntime) Generate(ctx context.Context, request ReferenceEditRequest) (EncodedVideo, error) {
	sink, err := NewGIFEncoder(r.profile.SampleFPS, UnitPixels)
	if err != nil {
		return EncodedVideo{}, err
	}
	if _, err := r.runtime.Run(ctx, request.Source.Pixels, request.Condition.InitialNoise, sink.Add); err != nil {
		return EncodedVideo{}, err
	}
	video, err := sink.Finish()
	if err == nil {
		r.runs++
		video.ResidentRun = r.runs
	}
	return video, err
}

func (r *LiveEditRuntime) Close(context.Context) error {
	if r == nil || r.runtime == nil {
		return nil
	}
	err := r.runtime.Close()
	r.runtime = nil
	return err
}

func RegisterLiveEditRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *LiveEditRuntime) error {
	if model == nil || model.runtime == nil {
		return errors.New("latent video: incomplete LiveEdit runtime binding")
	}
	return registerReferenceEditRuntime(runtime, modelID, model.Generate)
}

func readProgramProfile(ctx context.Context, store artifact.Reader, program recipe.Program) (Profile, error) {
	profileID, ok := program.Definition().PrimaryDependency(recipe.DependencyProfile)
	if !ok {
		return Profile{}, errors.New("latent video: compiled recipe has no profile")
	}
	profile, err := ReadProfile(ctx, store, profileID)
	if err != nil {
		return Profile{}, err
	}
	if err := profile.validate(); err != nil {
		return Profile{}, err
	}
	return profile, nil
}
