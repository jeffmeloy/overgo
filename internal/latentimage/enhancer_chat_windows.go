//go:build windows

package latentimage

import (
	"context"
	"errors"
	"path/filepath"

	"overgo/internal/sampling"
)

// enhancerDirectory is the folder in a MOSS Ultra model that serves its text encoder file as the enhancer's chat
// model: Qwen3.5-9B's config for generation, MOSS-Image's tokenizer and chat template, and the file
// (moss-ultra-sections/enhancer-chat).
const enhancerDirectory = "enhancer"

// EnhancerSampling is the workflow's TextGenerate sampling: temperature 0.7, top-k 20, top-p 0.8, presence penalty
// 1.5, seed 0 for each generation, min-p 0 and repetition penalty 1, in the order ComfyUI's processed_probs applies
// them (penalties, temperature, top-k, min-p, top-p), the penalties over every token the generation writes
// (RepeatLastN -1, read from the prompt's end: GenerateOptions.PenaltiesFrom).
func EnhancerSampling() sampling.Config {
	return sampling.Config{Temperature: 0.7, TopK: 20, TopP: 0.8, RepeatLastN: -1, RepeatPenalty: 1, PresencePenalty: 1.5, Seed: 0,
		Samplers: []sampling.SamplerStage{sampling.SamplerPenalties, sampling.SamplerTemperature, sampling.SamplerTopK, sampling.SamplerMinP, sampling.SamplerTopP}}
}

// PythonSpace is a rune Python's str.strip removes, as the enhancer's answers are trimmed.
func PythonSpace(r rune) bool { return pythonSpace(r) }

// EnhancerOpener opens the chat a MOSS model's enhancer folder serves (enhancerchat.Open), sampled as EnhancerSampling
// and trimmed as PythonSpace, and how it is closed. The picture package takes it from its caller, so no program drawing a
// picture links the language runtime a chat needs (simplediffusion-peak).
type EnhancerOpener func(ctx context.Context, directory string) (chat func(ctx context.Context, system, request string, maxTokens int) (string, error), close func() error, err error)

// EnhanceRequest is a request that asks prompt mode 2 (its Enhance) as the request stating what mode 2 writes: its
// prompt the source idea and its negative the exclusions the model's enhancer chat writes a positive and a negative of
// (enhance), the chat opened by open from the model's enhancer folder and closed before the picture is drawn. A request
// that asks no enhancing is returned as it is. A request the picture session would refuse once enhanced is refused
// before the chat opens, as the session refuses it: a request of -1 steps had the 9B chat opened and asked, its refusal
// coming after, or masked where the chat would not open (SQA-413).
func EnhanceRequest(ctx context.Context, open EnhancerOpener, modelDir string, request Request) (_ Request, err error) {
	if !request.Enhance {
		return request, nil
	}
	if request.Layout != nil {
		return Request{}, errEnhanceBesideLayout
	}
	enhanced := request
	enhanced.Enhance = false
	if err := ValidateRequest(enhanced); err != nil {
		return Request{}, err
	}
	chat, closeChat, err := open(ctx, filepath.Join(modelDir, enhancerDirectory))
	if err != nil {
		return Request{}, err
	}
	defer func() { err = errors.Join(err, closeChat()) }()
	positive, negative, err := enhance(ctx, chat, request.Prompt, request.NegativePrompt)
	if err != nil {
		return Request{}, err
	}
	request.Prompt, request.NegativePrompt, request.Enhance = positive, negative, false
	return request, nil
}
