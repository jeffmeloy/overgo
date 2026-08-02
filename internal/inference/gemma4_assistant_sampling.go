package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Gemma4AssistantSampledDraft: proposals plus sampler checkpoints.
type Gemma4AssistantSampledDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Distributions [][]sampling.TokenProbability
	History       []tokenizer.TokenID
	Base          *Gemma4AssistantSession
	samplerStates [][]byte
}

// DraftGemma4AssistantSampled: transactional sampled proposals.
func (r *Runner) DraftGemma4AssistantSampled(
	ctx context.Context,
	target *Runner,
	session *Gemma4AssistantSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *Gemma4AssistantSampledDraft, err error) {
	if r == nil || target == nil || session == nil || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: Gemma 4 assistant sampled draft inputs are invalid")
	}
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Gemma 4 assistant sampled draft limits are invalid")
	}
	initialState, err := sampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Gemma 4 assistant draft sampler: %w", err)
	}
	defer func() {
		restoreErr := sampler.LoadState(initialState)
		if err == nil && restoreErr != nil {
			draft = nil
			err = fmt.Errorf("inference: restore Gemma 4 assistant draft sampler: %w", restoreErr)
		}
	}()
	initialToken := history[len(history)-1]
	draft = &Gemma4AssistantSampledDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum),
		Distributions: make([][]sampling.TokenProbability, 0, maximum),
		History:       append([]tokenizer.TokenID(nil), history...), Base: session,
		samplerStates: [][]byte{append([]byte(nil), initialState...)},
	}
	currentToken, currentSession := initialToken, session
	currentHistory := tokenIDsAsInts(history)
	for range maximum {
		logits, next, advanceErr := r.AdvanceGemma4Assistant(ctx, target, currentToken, currentSession)
		if advanceErr != nil {
			return nil, advanceErr
		}
		result, sampleErr := sampler.SampleWithHistoryProbabilities(logits.Data, currentHistory, len(logits.Data))
		if sampleErr != nil {
			return nil, sampleErr
		}
		if result.SelectedProbability < minimumProbability {
			break
		}
		state, stateErr := sampler.SaveState()
		if stateErr != nil {
			return nil, fmt.Errorf("inference: save Gemma 4 assistant draft checkpoint: %w", stateErr)
		}
		token := tokenizer.TokenID(result.Token)
		draft.Tokens = append(draft.Tokens, token)
		draft.Probabilities = append(draft.Probabilities, result.SelectedProbability)
		draft.Distributions = append(draft.Distributions, append([]sampling.TokenProbability(nil), result.Top...))
		draft.samplerStates = append(draft.samplerStates, state)
		currentToken, currentSession = token, next
		currentHistory = append(currentHistory, result.Token)
		if target.vocab.IsEOG(token) {
			break
		}
	}
	return draft, nil
}

// VerifyGemma4AssistantSampled: ratio verification plus rollback.
func (r *Runner) VerifyGemma4AssistantSampled(
	ctx context.Context,
	target *Runner,
	draft *Gemma4AssistantSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *Gemma4AssistantVerification, err error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil || draft.Base.TargetCache == nil ||
		draftSampler == nil || targetSampler == nil || draftSampler == targetSampler {
		return nil, errors.New("inference: Gemma 4 assistant sampled verification inputs are invalid")
	}
	if len(draft.History) == 0 || draft.History[len(draft.History)-1] != draft.InitialToken ||
		len(draft.Tokens) != len(draft.Probabilities) || len(draft.Tokens) != len(draft.Distributions) ||
		len(draft.samplerStates) != len(draft.Tokens)+1 {
		return nil, errors.New("inference: Gemma 4 assistant sampled draft state is inconsistent")
	}
	if err := r.validateGemma4AssistantTarget(target); err != nil {
		return nil, err
	}
	draftOriginal, err := draftSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Gemma 4 assistant draft sampler: %w", err)
	}
	targetOriginal, err := targetSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Gemma 4 assistant target sampler: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = draftSampler.LoadState(draftOriginal)
			_ = targetSampler.LoadState(targetOriginal)
		}
	}()
	if err := draftSampler.LoadState(draft.samplerStates[0]); err != nil {
		return nil, fmt.Errorf("inference: restore Gemma 4 assistant draft base: %w", err)
	}
	assistantSession := draft.Base
	targetCache := draft.Base.TargetCache
	currentToken := draft.InitialToken
	history := tokenIDsAsInts(draft.History)
	accepted := 0
	for {
		logits, nextSession, advanceErr := r.advanceGemma4AssistantVerification(
			ctx, target, currentToken, assistantSession, targetCache,
		)
		if advanceErr != nil {
			return nil, advanceErr
		}
		assistantSession, targetCache = nextSession, nextSession.TargetCache
		var nextToken tokenizer.TokenID
		if accepted < len(draft.Tokens) {
			result, sampleErr := targetSampler.SpeculativeSample(
				logits.Data, history, draft.Distributions[accepted], int(draft.Tokens[accepted]),
			)
			if sampleErr != nil {
				return nil, sampleErr
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
				return nil, sampleErr
			}
			nextToken = tokenizer.TokenID(token)
		}
		if err := draftSampler.LoadState(draft.samplerStates[accepted]); err != nil {
			return nil, fmt.Errorf("inference: restore Gemma 4 assistant accepted draft: %w", err)
		}
		if err := draftSampler.AcceptToken(int(nextToken)); err != nil {
			return nil, fmt.Errorf("inference: commit Gemma 4 assistant target token to draft sampler: %w", err)
		}
		verification = &Gemma4AssistantVerification{
			Accepted: accepted, NextToken: nextToken, TargetLogits: logits, Session: assistantSession,
		}
		committed = true
		return verification, nil
	}
}

func (r *Runner) advanceGemma4AssistantVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	assistantSession *Gemma4AssistantSession,
	targetCache *KVCache,
) (reference.Value, *Gemma4AssistantSession, error) {
	hidden, nextTargetCache, err := target.ForwardCached(ctx, []tokenizer.TokenID{currentToken}, targetCache)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, err := target.projectHiddenLogits(ctx, hidden)
	if err != nil {
		return reference.Value{}, nil, err
	}
	_, nextSession, err := r.AdvanceGemma4Assistant(ctx, target, currentToken, assistantSession)
	if err != nil {
		return reference.Value{}, nil, err
	}
	width := int(hidden.Shape.Dims[0])
	nextSession.TargetCache = nextTargetCache
	nextSession.PendingHidden = reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), hidden.Data[len(hidden.Data)-width:]...),
	}
	nextSession.Position = effectiveCachePosition(nextTargetCache)
	return logits, nextSession, nil
}
