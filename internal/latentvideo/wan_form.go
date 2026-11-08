package latentvideo

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/sampling"
)

// referenceEncoderConfig carries the two published umT5 encoder-config
// facts the Wan checkpoint cannot carry (relative_attention_max_distance
// and layer_norm_epsilon); the encoder's geometry derives from its tensor
// shapes.
var referenceEncoderConfig = encoderConfig{RelativeMaxDistance: 128, NormEps: 1e-6}

// referenceNegativePrompt is the reference implementation's default
// negative prompt (wan/configs/shared_config.py, sample_neg_prompt): the
// unconditional branch of a prompt-form request that names no negative
// prompt of its own.
const referenceNegativePrompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

// The reference artifact names inside a Wan model directory that the text
// pipeline reads beside the denoiser: the umT5 encoder checkpoint and the
// HF tokenizer directory.
const (
	wanEncoderCheckpoint = "models_t5_umt5-xxl-enc-bf16.pth"
	wanTokenizerDir      = "google/umt5-xxl"
	// tokenizerTable and tokenizerRoles are the two files of the tokenizer
	// directory the pipeline reads: the unigram table and its special tokens.
	tokenizerTable = "tokenizer.json"
	tokenizerRoles = "special_tokens_map.json"
)

// TextPipelineIdentity names the bytes of the files a video model's text
// pipeline reads that the model's inventory does not hold: the umT5 encoder
// checkpoint and the tokenizer's two files, in the Wan directory the model
// conditions through, which is its own or, for a LiveEdit model, the base
// beside it (TextConditioningSpec.content). A stored answer is kept under it
// as well as under the model (capabilityruntime.Reading), so a changed
// encoder or tokenizer asks again. A file that is not there is named so,
// since a request that brings its own contexts is served without it. The
// checkpoint is named through the store's identity memo, which keeps what a
// process reads of it for the next (6.6 to 11.2 s of a LiveEdit first output
// when every process read it).
func TextPipelineIdentity(ctx context.Context, store artifact.Repository, modelDirectory string) (string, error) {
	if IsLiveEdit(modelDirectory) {
		modelDirectory, _ = LiveEditArtifacts(modelDirectory)
	}
	if IsWanStudent(modelDirectory) {
		student, err := resolveWanStudent(modelDirectory)
		if err != nil {
			return "", err
		}
		modelDirectory = student.base
	}
	var names *discovery.Memo
	if store != nil {
		names = discovery.LoadMemo(ctx, store)
	}
	identity, _, err := WanTextConditioningSpec(modelDirectory, 1).content(true, names)
	if err != nil {
		return "", err
	}
	if err := discovery.PublishMemo(ctx, store, names); err != nil && !errors.Is(err, overgodb.ErrReadOnly) {
		return "", err
	}
	return identity, nil
}

// WanTextConditioningSpec locates the text pipeline inside a Wan model
// directory at the denoiser's context length: the tokenizer, the encoder
// checkpoint, and the denoiser's own text projection.
func WanTextConditioningSpec(modelDirectory string, textLen int) TextConditioningSpec {
	return TextConditioningSpec{
		TokenizerDir:        filepath.Join(modelDirectory, filepath.FromSlash(wanTokenizerDir)),
		EncoderCheckpoint:   filepath.Join(modelDirectory, wanEncoderCheckpoint),
		ProjectionDir:       modelDirectory,
		SequenceLength:      textLen,
		RelativeMaxDistance: referenceEncoderConfig.RelativeMaxDistance,
		NormEps:             referenceEncoderConfig.NormEps,
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
// seed on its device. Either form names a clip format or leaves it to GIF.
func ValidateWanRequestForm(request WanRequest) error {
	if err := request.Format.validate(); err != nil {
		return err
	}
	if !request.PromptForm() {
		return validateWanRequest(request)
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
		request.NegativePrompt = referenceNegativePrompt
	}
	return request
}

// wanContextsHeld is how many context values the Wan text cache holds beside its kept prompts: about thirty contexts
// of 512 tokens at umT5's 4,096, a quarter of a gigabyte of host memory.
const wanContextsHeld = 1 << 26

// wanConditioning keeps prompts' contexts in the shared text cache (media.ConditioningCache) under the bytes they were
// made from, so a repeated prompt, or the standard negative prompt beside a new one, does not stream the CPU encoder
// again, and a prompt after a changed encoder or tokenizer does.
type wanConditioning struct {
	cache media.ConditioningCache
}

// contexts gives a prompt-form request's contexts: the cache's for each prompt it holds under the pipeline's bytes,
// and the pipeline's own for the others, which the cache then holds.
func (memo *wanConditioning) contexts(ctx context.Context, spec TextConditioningSpec, request WanRequest) (cond, uncond []float32, err error) {
	naming := processmeasure.Stage(ctx, "text pipeline named")
	content, _, err := spec.content(false, nil)
	naming()
	if err != nil {
		return nil, nil, err
	}
	contexts, err := memo.recall(content, []string{request.Prompt, request.NegativePrompt}, func(prompts []string) ([][]float32, error) {
		return promptContexts(ctx, spec, prompts)
	})
	if err != nil {
		return nil, nil, err
	}
	return contexts[0], contexts[1], nil
}

// recall gives each prompt's context under the encoder: the cache's where it holds one, and otherwise what encode
// makes of the prompts it does not hold, in one pass, which the cache then holds. The empty and the standard negative
// prompts are kept.
func (memo *wanConditioning) recall(encoder string, prompts []string, encode func([]string) ([][]float32, error)) ([][]float32, error) {
	if memo.cache.Limit == 0 {
		memo.cache = media.ConditioningCache{Limit: wanContextsHeld, Kept: []string{"", referenceNegativePrompt}}
	}
	contexts := make([][]float32, len(prompts))
	var missing []string
	for index, prompt := range prompts {
		if held, ok := memo.cache.Lookup(encoder, prompt); ok {
			contexts[index] = held
		} else if !slices.Contains(missing, prompt) {
			missing = append(missing, prompt)
		}
	}
	if len(missing) == 0 {
		return contexts, nil
	}
	encoded, err := encode(missing)
	if err != nil {
		return nil, err
	}
	for index, prompt := range prompts {
		if contexts[index] == nil {
			contexts[index] = encoded[slices.Index(missing, prompt)]
		}
	}
	for position, prompt := range missing {
		memo.cache.Store(encoder, prompt, encoded[position])
	}
	return contexts, nil
}

// promptContexts derives the contexts of prompts through the model's text pipeline, reading the projection once for
// all of them.
func promptContexts(ctx context.Context, spec TextConditioningSpec, prompts []string) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	weights, sourceBytes, err := loadProjectionWeights(spec.ProjectionDir)
	if err != nil {
		return nil, err
	}
	conditioned, err := textConditioningsWithWeights(ctx, spec, weights, sourceBytes, prompts...)
	if err != nil {
		return nil, err
	}
	contexts := make([][]float32, len(conditioned))
	for index, result := range conditioned {
		contexts[index] = result.Context
	}
	return contexts, nil
}
