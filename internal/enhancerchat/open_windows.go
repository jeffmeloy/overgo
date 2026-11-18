//go:build windows

package enhancerchat

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
)

// Open opens the chat model the folder at directory serves, as overgo serves a Hugging Face directory it has not
// published (modelintake.PrepareInferenceCandidate), and the chat its runner answers with: the request in the chat
// template under the system prompt with thinking off, sampled by config with a sampler of its own for each answer,
// decoded without its special tokens and trimmed of the runes trim names; close closes the runner.
func Open(ctx context.Context, store artifact.Reader, directory string, config sampling.Config, trim func(rune) bool) (chat Chat, close func() error, err error) {
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
	chat = func(ctx context.Context, system, request string, maxTokens int) (string, error) {
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
		sampler, err := sampling.New(config)
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
		return strings.TrimFunc(answer, trim), nil
	}
	return chat, runner.Close, nil
}
