//go:build windows

package latentimage

import (
	"context"
	"errors"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
)

// enhancerDirectory is the folder in a MOSS Ultra model that serves its text encoder file as the enhancer's chat
// model: Qwen3.5-9B's config for generation, MOSS-Image's tokenizer and chat template, and the file
// (moss-ultra-sections/enhancer-chat).
const enhancerDirectory = "enhancer"

// enhancerSampling is the workflow's TextGenerate sampling: temperature 0.7, top-k 20, top-p 0.8, presence penalty
// 1.5, seed 0 for each generation, min-p 0 and repetition penalty 1, in the order ComfyUI's processed_probs applies
// them (penalties, temperature, top-k, min-p, top-p), the penalties over every token the generation writes
// (RepeatLastN -1, read from the prompt's end: GenerateOptions.PenaltiesFrom).
var enhancerSampling = sampling.Config{Temperature: 0.7, TopK: 20, TopP: 0.8, RepeatLastN: -1, RepeatPenalty: 1, PresencePenalty: 1.5, Seed: 0,
	Samplers: []sampling.SamplerStage{sampling.SamplerPenalties, sampling.SamplerTemperature, sampling.SamplerTopK, sampling.SamplerMinP, sampling.SamplerTopP}}

// EnhanceRequest is a request that asks prompt mode 2 (its Enhance) as the request stating what mode 2 writes: its
// prompt the source idea and its negative the exclusions the model's enhancer chat writes a positive and a negative of
// (enhance), the chat opened from the model's enhancer folder through the store's profiles and closed before the
// picture is drawn. A request that asks no enhancing is returned as it is.
func EnhanceRequest(ctx context.Context, store artifact.Reader, modelDir string, request Request) (_ Request, err error) {
	if !request.Enhance {
		return request, nil
	}
	if request.Layout != nil {
		return Request{}, errEnhanceBesideLayout
	}
	chat, closeChat, err := openEnhancerChat(ctx, store, filepath.Join(modelDir, enhancerDirectory))
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

// openEnhancerChat opens the chat model the directory serves, as overgo serves a Hugging Face directory it has not
// published (modelintake.PrepareInferenceCandidate), and the chat its runner answers with: the request in the chat
// template under the system prompt with thinking off, sampled as TextGenerate samples, its answer decoded without its
// special tokens and trimmed.
func openEnhancerChat(ctx context.Context, store artifact.Reader, directory string) (enhancerChat, func() error, error) {
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, store, directory, modelintake.SessionOverride{}, recipe.ResidencyDeviceNative)
	if err != nil {
		return nil, nil, err
	}
	loaded, err := modelrecipe.ResolveCandidateGGUF(ctx, store, directory, candidate.Definition, candidate.Resolved, candidate.Inventory.Claim)
	if err != nil {
		return nil, nil, err
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		return nil, nil, errors.Join(err, loaded.Close())
	}
	chat := func(ctx context.Context, system, request string, maxTokens int) (string, error) {
		text, err := runner.FormatChatWithOptions([]inference.ChatMessage{
			{Role: inference.ChatRoleSystem, Content: system}, {Role: inference.ChatRoleUser, Content: request},
		}, inference.ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: false})
		if err != nil {
			return "", err
		}
		prompt, err := runner.TokenizeText(text, false, true)
		if err != nil {
			return "", err
		}
		// Each TextGenerate seeds its own sampler.
		sampler, err := sampling.New(enhancerSampling)
		if err != nil {
			return "", err
		}
		ids, _, err := runner.Generate(ctx, "", inference.GenerateOptions{PromptTokenIDs: prompt, PenaltiesFrom: len(prompt), MaxNewTokens: maxTokens, Sampler: sampler})
		if err != nil {
			return "", err
		}
		answer, err := runner.Vocab().Decode(ids[len(prompt):], false)
		if err != nil {
			return "", err
		}
		return strings.TrimFunc(answer, pythonSpace), nil
	}
	return chat, runner.Close, nil
}
