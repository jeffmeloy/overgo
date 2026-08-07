package inference

import (
	"fmt"
	"math"
	"slices"

	"overgo/internal/sampling"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func validSampledLimits(maximum int, minimumProbability float64) bool {
	return maximum > 0 && minimumProbability >= 0 && minimumProbability <= 1 && !math.IsNaN(minimumProbability)
}

func validVerificationSamplers(draft, target *sampling.Sampler) bool {
	return draft != nil && target != nil && draft != target
}

type sampledDraft[S any] struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Distributions [][]sampling.TokenProbability
	History       []tokenizer.TokenID
	Base          S
	samplerStates [][]byte
}

func draftSampled[S any](
	base S,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
	label string,
	isEOG func(tokenizer.TokenID) bool,
	advance func(tokenizer.TokenID, S, int) ([]float32, S, error),
) (draft *sampledDraft[S], err error) {
	initialState, err := sampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save %s draft sampler: %w", label, err)
	}
	defer func() {
		restoreErr := sampler.LoadState(initialState)
		if err == nil && restoreErr != nil {
			draft = nil
			err = fmt.Errorf("inference: restore %s draft sampler: %w", label, restoreErr)
		}
	}()
	initialToken := history[len(history)-1]
	draft = &sampledDraft[S]{
		InitialToken:  initialToken,
		Tokens:        make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum),
		Distributions: make([][]sampling.TokenProbability, 0, maximum),
		History:       slices.Clone(history),
		Base:          base,
		samplerStates: [][]byte{slices.Clone(initialState)},
	}
	currentToken, currentState := initialToken, base
	currentHistory := tokenHistory(history)
	for index := range maximum {
		logits, next, advanceErr := advance(currentToken, currentState, index)
		if advanceErr != nil {
			return nil, advanceErr
		}
		result, sampleErr := sampler.SampleWithHistoryProbabilities(logits, currentHistory, len(logits))
		if sampleErr != nil {
			return nil, sampleErr
		}
		if result.SelectedProbability < minimumProbability {
			break
		}
		state, stateErr := sampler.SaveState()
		if stateErr != nil {
			return nil, fmt.Errorf("inference: save %s draft checkpoint: %w", label, stateErr)
		}
		token := tokenizer.TokenID(result.Token)
		draft.Tokens = append(draft.Tokens, token)
		draft.Probabilities = append(draft.Probabilities, result.SelectedProbability)
		draft.Distributions = append(draft.Distributions, slices.Clone(result.Top))
		draft.samplerStates = append(draft.samplerStates, state)
		currentToken, currentState = token, next
		currentHistory = append(currentHistory, result.Token)
		if isEOG(token) {
			break
		}
	}
	return draft, nil
}

func validSampledDraft[S any](draft *sampledDraft[S]) bool {
	return draft != nil && len(draft.History) > 0 &&
		draft.History[len(draft.History)-1] == draft.InitialToken &&
		len(draft.Tokens) == len(draft.Probabilities) &&
		len(draft.Tokens) == len(draft.Distributions) &&
		len(draft.samplerStates) == len(draft.Tokens)+1
}

func verifySampled[D, S, V any](
	draft *sampledDraft[D],
	initial S,
	draftSampler, targetSampler *sampling.Sampler,
	label string,
	advance func(tokenizer.TokenID, S) (reference.Value, S, error),
	finish func(int, tokenizer.TokenID, reference.Value, S) (V, error),
) (verification V, err error) {
	draftOriginal, err := draftSampler.SaveState()
	if err != nil {
		return verification, fmt.Errorf("inference: save %s draft sampler: %w", label, err)
	}
	targetOriginal, err := targetSampler.SaveState()
	if err != nil {
		return verification, fmt.Errorf("inference: save %s target sampler: %w", label, err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = draftSampler.LoadState(draftOriginal)
			_ = targetSampler.LoadState(targetOriginal)
		}
	}()
	if err := draftSampler.LoadState(draft.samplerStates[0]); err != nil {
		return verification, fmt.Errorf("inference: restore %s draft base: %w", label, err)
	}
	state := initial
	currentToken := draft.InitialToken
	history := tokenHistory(draft.History)
	accepted := 0
	for {
		logits, next, advanceErr := advance(currentToken, state)
		if advanceErr != nil {
			return verification, advanceErr
		}
		state = next
		var nextToken tokenizer.TokenID
		if accepted < len(draft.Tokens) {
			result, sampleErr := targetSampler.SpeculativeSample(
				logits.Data, history, draft.Distributions[accepted], int(draft.Tokens[accepted]),
			)
			if sampleErr != nil {
				return verification, sampleErr
			}
			nextToken = tokenizer.TokenID(result.Token)
			if result.Accepted {
				accepted++
				currentToken = nextToken
				history = append(history, result.Token)
				continue
			}
		} else {
			token, sampleErr := targetSampler.SampleWithHistory(logits.Data, history)
			if sampleErr != nil {
				return verification, sampleErr
			}
			nextToken = tokenizer.TokenID(token)
		}
		if err := draftSampler.LoadState(draft.samplerStates[accepted]); err != nil {
			return verification, fmt.Errorf("inference: restore %s accepted draft: %w", label, err)
		}
		if err := draftSampler.AcceptToken(int(nextToken)); err != nil {
			return verification, fmt.Errorf("inference: commit %s target token to draft sampler: %w", label, err)
		}
		verification, err = finish(accepted, nextToken, logits, state)
		if err != nil {
			return verification, err
		}
		committed = true
		return verification, nil
	}
}
