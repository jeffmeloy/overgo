//go:build windows

package latentvideo

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/media"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

// chunkCausalSampler names the sampler a chunk-causal model's clip ran, as its preset records it.
const chunkCausalSampler = "chunk-causal flow"

// WanStudentRuntime serves a few-step student of Wan as a text-to-video model: a prompt-form Wan request runs through
// the chunk-causal runtime under the student's policy, its prompt read by the base's text encoder on the card and
// projected by the student's own projection. It is compiled for one clip's geometry, and a request of another is
// refused at Reset, so its director loads one for it.
type WanStudentRuntime struct {
	runtime         *referenceEditRuntime
	profile         Profile
	text            TextConditioningSpec
	encoder         string
	projection      projectionWeights
	projectionBytes int64
	memo            wanConditioning
	runs            int
}

// LoadWanStudentRuntime loads the student in path for the clip a request asks, its omitted parameters the profile's.
func LoadWanStudentRuntime(ctx context.Context, store artifact.Repository, path string, program recipe.Program, request WanRequest) (*WanStudentRuntime, error) {
	profile, err := readProgramProfile(ctx, store, program)
	if err != nil {
		return nil, err
	}
	student, err := resolveWanStudent(path)
	if err != nil {
		return nil, err
	}
	request = request.withGeneration(profile.Generation)
	volume, err := media.DownsampledVolume(request.Frames, request.Height, request.Width, profile.Policy.VAEStride)
	if err != nil {
		return nil, err
	}
	if err := student.policy.admitLatentFrames(volume.Frames); err != nil {
		return nil, err
	}
	config, err := LoadDenoiserConfig(student.base, profile.Policy)
	if err != nil {
		return nil, err
	}
	schedule, err := chunkCausalSchedule(student.policy, profile)
	if err != nil {
		return nil, err
	}
	text := WanTextConditioningSpec(student.base, config.TextLen)
	text.Card = true
	named, _, err := text.content(false, nil)
	if err != nil {
		return nil, err
	}
	runtime, err := NewReferenceEditRuntime(ReferenceEditRuntimeConfig{
		WanDirectory: student.base, EditCheckpoint: student.checkpoint, Policy: profile.Policy, LatentStats: profile.LatentStats,
		Source:         SourceVideoShape{Channels: media.RGBChannels, Frames: request.Frames, Height: request.Height, Width: request.Width},
		FramesPerChunk: schedule.FramesPerChunk, LocalAttentionFrames: schedule.LocalAttention,
		Timesteps: schedule.Timesteps, Sigmas: schedule.Sigmas, ContextTimestep: schedule.ContextTimestep, Warp: schedule.Warp,
		Layers: config.NumLayers, DeviceOrdinal: device.DefaultOrdinal(), CodecLibrary: true,
	})
	if err != nil {
		return nil, err
	}
	projection, projectionBytes, err := runtime.checkpoint.loadTextProjection()
	if err != nil {
		return nil, errors.Join(err, runtime.Close())
	}
	// The student's projection is not the base's, so its contexts are kept under its checkpoint too.
	return &WanStudentRuntime{
		runtime: runtime, profile: profile, text: text, encoder: named + "\x00" + student.checkpoint,
		projection: projection, projectionBytes: projectionBytes,
	}, nil
}

// admit refuses a request the student does not run as it asks: one with no prompt, or with steps, a shift or guidance
// other than the student's own, which its distillation fixed.
func (r *WanStudentRuntime) admit(request WanRequest) error {
	generation := r.profile.Generation
	if !request.PromptForm() {
		return errors.New("latent video: a few-step student is asked by prompt")
	}
	if !checked.Equal(request.Steps, generation.Steps) || !checked.Equal(request.Shift, generation.Shift) || !checked.Equal(request.GuideScale, generation.GuideScale) {
		return fmt.Errorf("latent video: the student runs %d steps at shift %g unguided, asked %d at shift %g with guidance %g",
			generation.Steps, generation.Shift, request.Steps, request.Shift, request.GuideScale)
	}
	return nil
}

// Reset admits a request of the geometry the runtime was compiled for.
func (r *WanStudentRuntime) Reset(_ context.Context, request WanRequest) error {
	if r == nil || r.runtime == nil {
		return errors.New("latent video: student runtime is closed")
	}
	request = request.withGeneration(r.profile.Generation)
	volume, err := media.DownsampledVolume(request.Frames, request.Height, request.Width, r.profile.Policy.VAEStride)
	if err != nil {
		return err
	}
	latent := r.runtime.editPlan.Latent
	if !checked.Equal(media.VolumeGeometry{Frames: latent.LatentFrames, Height: latent.LatentHeight, Width: latent.LatentWidth}, volume) {
		return errors.New("latent video: request differs from the resident student's geometry")
	}
	return r.admit(request)
}

// Generate draws the clip a prompt-form request asks, the prompt's context the one kept for it where a request before
// asked it.
func (r *WanStudentRuntime) Generate(ctx context.Context, request WanRequest) (EncodedVideo, error) {
	if r == nil || r.runtime == nil {
		return EncodedVideo{}, errors.New("latent video: student runtime is closed")
	}
	request = request.withGeneration(r.profile.Generation)
	if err := r.admit(request); err != nil {
		return EncodedVideo{}, err
	}
	contexts, err := r.memo.recall(r.encoder, []string{request.Prompt}, func(prompts []string) ([][]float32, error) {
		conditioned, err := textConditioningsWithWeights(ctx, r.text, r.projection, r.projectionBytes, prompts...)
		contexts := make([][]float32, len(conditioned))
		for index, result := range conditioned {
			contexts[index] = result.Context
		}
		return contexts, err
	})
	if err != nil {
		return EncodedVideo{}, err
	}
	encoder, err := newServedEncoder(ctx, request.Format, r.profile.SampleFPS, SignedUnitPixels)
	if err != nil {
		return EncodedVideo{}, err
	}
	video, err := writeClip(ctx, encoder, func(sink VideoFrameSink) error {
		_, err := r.runtime.Run(ctx, contexts[0], int64(request.Seed), nil, nil, sink)
		return err
	})
	if err == nil {
		r.runs++
		video.ResidentRun = r.runs
		video.Preset = media.Preset{Width: request.Width, Height: request.Height, Frames: request.Frames, Steps: request.Steps,
			Guidance: request.GuideScale, Sampler: chunkCausalSampler, Shift: request.Shift}
	}
	return video, err
}

// PeakDeviceBytes reports the largest peak the resident device sessions record in their allocation ledgers.
func (r *WanStudentRuntime) PeakDeviceBytes() (uint64, error) {
	if r == nil || r.runtime == nil {
		return 0, errors.New("latent video: student runtime is closed")
	}
	return peakAcrossSessions(r.runtime.denoiser, r.runtime.decoder)
}

// Close gives the student's sessions back to the card; a closed runtime holds none.
func (r *WanStudentRuntime) Close(context.Context) error {
	if r == nil {
		return nil
	}
	held := r.runtime
	r.runtime = nil
	return held.Close()
}

// RegisterWanStudentRuntime binds the student's generation to the stages of the generation definition Wan has; a
// runtime closed or never loaded refuses each request (Generate).
func RegisterWanStudentRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *WanStudentRuntime) error {
	return registerWanRuntime(runtime, modelID, model.Generate)
}
