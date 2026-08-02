package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

// Eagle3SampledDraft: proposals plus sampler checkpoints.
type Eagle3SampledDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Distributions [][]sampling.TokenProbability
	History       []tokenizer.TokenID
	Base          *Eagle3Session
	samplerStates [][]byte
}

// DraftEagle3Sampled: transactional sampled proposals.
func (r *Runner) DraftEagle3Sampled(
	ctx context.Context,
	target *Runner,
	session *Eagle3Session,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *Eagle3SampledDraft, err error) {
	if r == nil || target == nil || session == nil || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: Eagle3 sampled draft inputs are invalid")
	}
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Eagle3 sampled draft limits are invalid")
	}
	initialState, err := sampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Eagle3 draft sampler: %w", err)
	}
	defer func() {
		restoreErr := sampler.LoadState(initialState)
		if err == nil && restoreErr != nil {
			draft = nil
			err = fmt.Errorf("inference: restore Eagle3 draft sampler: %w", restoreErr)
		}
	}()
	initialToken := history[len(history)-1]
	draft = &Eagle3SampledDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum),
		Distributions: make([][]sampling.TokenProbability, 0, maximum),
		History:       append([]tokenizer.TokenID(nil), history...), Base: session,
		samplerStates: [][]byte{append([]byte(nil), initialState...)},
	}
	currentToken, currentSession := initialToken, session
	currentHistory := tokenIDsAsInts(history)
	for range maximum {
		logits, next, advanceErr := r.AdvanceEagle3(ctx, target, currentToken, currentSession)
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
			return nil, fmt.Errorf("inference: save Eagle3 draft checkpoint: %w", stateErr)
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

// VerifyEagle3Sampled: ratio verification plus rollback.
func (r *Runner) VerifyEagle3Sampled(
	ctx context.Context,
	target *Runner,
	draft *Eagle3SampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *Eagle3Verification, err error) {
	if r == nil || target == nil || draft == nil || !validEagle3CoordinatorSession(draft.Base) ||
		draftSampler == nil || targetSampler == nil || draftSampler == targetSampler {
		return nil, errors.New("inference: Eagle3 sampled verification inputs are invalid")
	}
	if len(draft.History) == 0 || draft.History[len(draft.History)-1] != draft.InitialToken ||
		len(draft.Tokens) != len(draft.Probabilities) || len(draft.Tokens) != len(draft.Distributions) ||
		len(draft.samplerStates) != len(draft.Tokens)+1 {
		return nil, errors.New("inference: Eagle3 sampled draft state is inconsistent")
	}
	draftOriginal, err := draftSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Eagle3 draft sampler: %w", err)
	}
	targetOriginal, err := targetSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Eagle3 target sampler: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = draftSampler.LoadState(draftOriginal)
			_ = targetSampler.LoadState(targetOriginal)
		}
	}()
	if err := draftSampler.LoadState(draft.samplerStates[0]); err != nil {
		return nil, fmt.Errorf("inference: restore Eagle3 draft base: %w", err)
	}
	session := draft.Base
	currentToken := draft.InitialToken
	history := tokenIDsAsInts(draft.History)
	accepted := 0
	for {
		logits, next, advanceErr := r.advanceEagle3Verification(ctx, target, currentToken, session)
		if advanceErr != nil {
			return nil, advanceErr
		}
		session = next
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
			return nil, fmt.Errorf("inference: restore Eagle3 accepted draft: %w", err)
		}
		if err := draftSampler.AcceptToken(int(nextToken)); err != nil {
			return nil, fmt.Errorf("inference: commit Eagle3 target token to draft sampler: %w", err)
		}
		verification = &Eagle3Verification{
			Accepted: accepted, NextToken: nextToken, TargetLogits: logits, Session: session,
		}
		committed = true
		return verification, nil
	}
}
