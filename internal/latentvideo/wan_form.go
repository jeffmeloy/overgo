package latentvideo

import (
	"context"
	"errors"
	"math"
	"path/filepath"

	"overgo/internal/sampling"
)

// ReferenceEncoderConfig carries the two published umT5 encoder-config
// facts the Wan checkpoint cannot carry (relative_attention_max_distance
// and layer_norm_epsilon); the encoder's geometry derives from its tensor
// shapes.
var ReferenceEncoderConfig = EncoderConfig{RelativeMaxDistance: 128, NormEps: 1e-6}

// ReferenceNegativePrompt is the reference implementation's default
// negative prompt (wan/configs/shared_config.py, sample_neg_prompt): the
// unconditional branch of a prompt-form request that names no negative
// prompt of its own.
const ReferenceNegativePrompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

// The reference artifact names inside a Wan model directory that the text
// pipeline reads beside the denoiser: the umT5 encoder checkpoint and the
// HF tokenizer directory.
const (
	wanEncoderCheckpoint = "models_t5_umt5-xxl-enc-bf16.pth"
	wanTokenizerDir      = "google/umt5-xxl"
)

// WanTextConditioningSpec locates the text pipeline inside a Wan model
// directory at the denoiser's context length: the tokenizer, the encoder
// checkpoint, and the denoiser's own text projection.
func WanTextConditioningSpec(modelDirectory string, textLen int) TextConditioningSpec {
	return TextConditioningSpec{
		TokenizerDir:        filepath.Join(modelDirectory, filepath.FromSlash(wanTokenizerDir)),
		EncoderCheckpoint:   filepath.Join(modelDirectory, wanEncoderCheckpoint),
		ProjectionDir:       modelDirectory,
		SequenceLength:      textLen,
		RelativeMaxDistance: ReferenceEncoderConfig.RelativeMaxDistance,
		NormEps:             ReferenceEncoderConfig.NormEps,
	}
}

// PromptForm reports whether the request conditions through prompts
// rather than resident contexts: the page form, which the runtime
// resolves through the model's own text pipeline.
func (request WanRequest) PromptForm() bool {
	return request.Prompt != "" && len(request.CondContext) == 0 && len(request.UncondContext) == 0
}

// ValidateWanRequestForm admits a complete tensor-form request or the
// prompt form: a prompt without contexts, each generation parameter either
// given or left at zero for the profile's generation policy, and neither
// an initial sample nor a noise plan, which the runtime compiles from the
// seed on its device.
func ValidateWanRequestForm(request WanRequest) error {
	if !request.PromptForm() {
		return ValidateWanRequest(request)
	}
	for _, value := range []int{request.Frames, request.Width, request.Height, request.Steps} {
		if value < 0 {
			return errors.New("latent video: negative Wan generation parameter")
		}
	}
	for _, value := range []float64{request.Shift, request.GuideScale} {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return errors.New("latent video: invalid Wan generation parameter")
		}
	}
	if len(request.InitialSample) != 0 || request.Noise != (sampling.CounterNoisePlan{}) {
		return errors.New("latent video: a prompt-form Wan request carries no sample or noise plan")
	}
	return nil
}

// withGeneration fills every generation parameter the request leaves at
// zero from the profile's generation policy, and a prompt-form request's
// missing negative prompt from the reference default; a page request
// names only what it changes.
func (request WanRequest) withGeneration(policy generationPolicy) WanRequest {
	if request.Frames == 0 {
		request.Frames = policy.Frames
	}
	if request.Width == 0 {
		request.Width = policy.Width
	}
	if request.Height == 0 {
		request.Height = policy.Height
	}
	if request.Steps == 0 {
		request.Steps = policy.Steps
	}
	if request.Shift == 0 {
		request.Shift = policy.Shift
	}
	if request.GuideScale == 0 {
		request.GuideScale = policy.GuideScale
	}
	if request.Seed == 0 {
		request.Seed = policy.Seed
	}
	if request.PromptForm() && request.NegativePrompt == "" {
		request.NegativePrompt = ReferenceNegativePrompt
	}
	return request
}

// promptContexts derives the conditional and unconditional contexts of a
// prompt-form request through the model's text pipeline, reading the
// projection once for both branches.
func promptContexts(ctx context.Context, spec TextConditioningSpec, request WanRequest) (cond, uncond []float32, err error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	weights, sourceBytes, err := loadProjectionWeights(spec.ProjectionDir)
	if err != nil {
		return nil, nil, err
	}
	conditional, err := textConditioningWithWeights(ctx, spec, request.Prompt, weights, sourceBytes)
	if err != nil {
		return nil, nil, err
	}
	unconditional, err := textConditioningWithWeights(ctx, spec, request.NegativePrompt, weights, sourceBytes)
	if err != nil {
		return nil, nil, err
	}
	return conditional.Context, unconditional.Context, nil
}
