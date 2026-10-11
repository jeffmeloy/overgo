//go:build windows

package latentvideo

import (
	"context"
	"errors"
	"reflect"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/inferencesurface"
	"overgo/internal/media"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tensor/dtype"
	"overgo/internal/workflowruntime"
)

// WanRuntime retains one generator across compatible requests, the text
// pipeline that resolves a prompt-form request, the device occupancy its
// noise plans compile against, and the last prompt pair's contexts.
type WanRuntime struct {
	generator *Generator
	profile   Profile
	text      TextConditioningSpec
	occupancy editNoiseStream
	memo      wanConditioning
	runs      int
}

// wanConditioning memoizes the contexts of one prompt pair, so a replayed
// or repeated request does not stream the encoder again.
type wanConditioning struct {
	prompt, negative string
	cond, uncond     []float32
}

// LoadWanRuntime loads the generator a request is served by. A request that
// names a prompt has its prompts encoded on the host while the generator
// loads on the device, since neither waits on the other: the encoder keeps
// their rows, and the request finds them when it resolves. The rows of the
// negative prompt every such request carries are kept in the store, under
// the identity of this package's code where the build counted one, so that a
// process encodes that prompt only when no process before it has.
func LoadWanRuntime(ctx context.Context, store artifact.Repository, path string, program recipe.Program, request WanRequest) (*WanRuntime, error) {
	profile, err := readProgramProfile(ctx, store, program)
	if err != nil {
		return nil, err
	}
	request = request.withGeneration(profile.Generation)
	config, err := LoadDenoiserConfig(path, profile.Policy)
	if err != nil {
		return nil, err
	}
	text := WanTextConditioningSpec(path, config.TextLen)
	if identity, counted := inferencesurface.Execution(reflect.TypeFor[WanRuntime]().PkgPath()); counted {
		text.Common, text.Kept, text.KeptUnder = referenceNegativePrompt, store, identity
	}
	if request.PromptForm() {
		// What the pass finds wrong, the request finds again when it resolves, and says.
		go encodePrompts(processmeasure.Unstaged(ctx), text, request.Prompt, request.NegativePrompt)
	}
	storage := dtype.BF16
	if !checked.Equal(profile.Precision.MatmulWeights, dtype.BF16.String()) {
		return nil, errors.New("latent video: unsupported production precision")
	}
	ordinal := device.DefaultOrdinal()
	generator, err := NewGenerator(GeneratorConfig{
		ModelDirectory: path, Policy: profile.Policy, LatentStats: profile.LatentStats,
		Frames: request.Frames, Width: request.Width, Height: request.Height,
		Precision: DenoiserPrecision{
			MatmulWeights: storage, RoundAttentionStorage: profile.Precision.RoundAttentionStorage,
		}, DeviceOrdinal: ordinal, CodecLibrary: true,
	})
	if err != nil {
		return nil, err
	}
	occupancy, err := newEditNoiseStream(0, ordinal)
	if err != nil {
		return nil, errors.Join(err, generator.Close())
	}
	return &WanRuntime{
		generator: generator, profile: profile,
		text: text, occupancy: occupancy,
	}, nil
}

// resolve completes a request against the resident runtime: the profile's
// generation policy fills omitted parameters, a prompt-form request gains
// its contexts from the text pipeline and its noise plan from the seed on
// this device, and the result is the complete request the generator runs.
func (r *WanRuntime) resolve(ctx context.Context, request WanRequest) (WanRequest, error) {
	if err := ctx.Err(); err != nil {
		return WanRequest{}, err
	}
	request = request.withGeneration(r.profile.Generation)
	if !request.PromptForm() {
		return request, validateWanRequest(request)
	}
	if r.memo.cond == nil || r.memo.prompt != request.Prompt || r.memo.negative != request.NegativePrompt {
		cond, uncond, err := promptContexts(ctx, r.text, request)
		if err != nil {
			return WanRequest{}, err
		}
		r.memo = wanConditioning{prompt: request.Prompt, negative: request.NegativePrompt, cond: cond, uncond: uncond}
	}
	request.CondContext, request.UncondContext = r.memo.cond, r.memo.uncond
	plan, _, err := sampling.CompileCounterNoisePlan(
		request.Seed, 0, r.generator.Geometry().Elements(), r.occupancy.smCount, r.occupancy.threadsPerSM,
	)
	if err != nil {
		return WanRequest{}, err
	}
	request.Noise = plan
	return request, validateWanRequest(request)
}

func (r *WanRuntime) Reset(_ context.Context, request WanRequest) error {
	if r == nil || r.generator == nil {
		return errors.New("latent video: Wan runtime is closed")
	}
	request = request.withGeneration(r.profile.Generation)
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
	request, err := r.resolve(ctx, request)
	if err != nil {
		return EncodedVideo{}, err
	}
	encoder, err := NewClipEncoder(r.profile.SampleFPS, SignedUnitPixels)
	if err != nil {
		return EncodedVideo{}, err
	}
	video, err := encoder.clip(ctx, func(sink VideoFrameSink) error {
		_, err := r.generator.Generate(ctx, GenerateRequest{
			Steps: request.Steps, Shift: request.Shift, GuideScale: request.GuideScale,
			CondContext: request.CondContext, UncondContext: request.UncondContext,
			InitialSample: request.InitialSample, Noise: request.Noise, Sink: sink,
		})
		return err
	})
	if err == nil {
		r.runs++
		video.ResidentRun = r.runs
	}
	return video, err
}

// PeakDeviceBytes reports the largest peak the resident device
// sessions record in their allocation ledgers.
func (r *WanRuntime) PeakDeviceBytes() (uint64, error) {
	if r == nil || r.generator == nil {
		return 0, errors.New("latent video: Wan runtime is closed")
	}
	return peakAcrossSessions(r.generator.denoiser, r.generator.decoder)
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
	runtime *referenceEditRuntime
	profile Profile
	runs    int
}

func LoadLiveEditRuntime(ctx context.Context, store artifact.Reader, path string, program recipe.Program, request ReferenceEditRequest) (*LiveEditRuntime, error) {
	profile, err := readProgramProfile(ctx, store, program)
	if err != nil {
		return nil, err
	}
	wanDirectory, editCheckpoint := LiveEditArtifacts(path)
	base, err := LoadDenoiserConfig(wanDirectory, profile.Policy)
	if err != nil {
		return nil, err
	}
	source := request.Source
	condition := request.Condition
	runtime, err := NewReferenceEditRuntime(ReferenceEditRuntimeConfig{
		WanDirectory: wanDirectory, EditCheckpoint: editCheckpoint,
		Policy: profile.Policy, LatentStats: profile.LatentStats,
		Source:         SourceVideoShape{Channels: source.Channels, Frames: source.Frames, Height: source.Height, Width: source.Width},
		FramesPerChunk: condition.FramesPerChunk, LocalAttentionFrames: condition.LocalAttention,
		Timesteps: condition.Timesteps, Sigmas: condition.Sigmas, ContextTimestep: condition.ContextTimestep,
		Layers: base.NumLayers, DeviceOrdinal: device.DefaultOrdinal(), CodecLibrary: true,
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

// Generate edits the request's source clip and returns the edited clip. The
// edit works in the range its source arrives in, 0 to 1 (sourceVideoFromGIF),
// so the frames it decodes are written as that range: read as -1 to 1 they
// come out washed, every black lifted to grey.
func (r *LiveEditRuntime) Generate(ctx context.Context, request ReferenceEditRequest) (EncodedVideo, error) {
	encoder, err := NewClipEncoder(r.profile.SampleFPS, UnitPixels)
	if err != nil {
		return EncodedVideo{}, err
	}
	condition := request.Condition
	video, err := encoder.clip(ctx, func(sink VideoFrameSink) error {
		_, err := r.runtime.Run(ctx, condition.TextContext, condition.Seed, request.Source.Pixels, condition.InitialNoise, sink)
		return err
	})
	if err == nil {
		r.runs++
		video.ResidentRun = r.runs
	}
	return video, err
}

// PeakDeviceBytes reports the largest peak the resident device
// sessions record in their allocation ledgers.
func (r *LiveEditRuntime) PeakDeviceBytes() (uint64, error) {
	if r == nil || r.runtime == nil {
		return 0, errors.New("latent video: LiveEdit runtime is closed")
	}
	return peakAcrossSessions(r.runtime.denoiser, r.runtime.decoder)
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

// peakAcrossSessions folds session allocation ledgers into the single
// largest observed peak; sessions run on distinct workers, so the
// maximum is the runtime's device high-water mark, not a sum.
func peakAcrossSessions(sessions ...interface {
	MemoryStats() (driver.MemoryStats, error)
}) (uint64, error) {
	var peak uint64
	var errs []error
	for _, session := range sessions {
		stats, err := session.MemoryStats()
		if err != nil {
			errs = append(errs, err)
			continue
		}
		peak = max(peak, stats.PeakBytes)
	}
	if peak == 0 && len(errs) > 0 {
		return 0, errors.Join(errs...)
	}
	return peak, nil
}

func readProgramProfile(ctx context.Context, store artifact.Reader, program recipe.Program) (Profile, error) {
	profileID, ok := program.Definition().PrimaryDependency(recipe.DependencyProfile)
	if !ok {
		return Profile{}, errors.New("latent video: compiled recipe has no profile")
	}
	profile, err := readProfile(ctx, store, profileID)
	if err != nil {
		return Profile{}, err
	}
	if err := profile.validate(); err != nil {
		return Profile{}, err
	}
	return profile, nil
}
