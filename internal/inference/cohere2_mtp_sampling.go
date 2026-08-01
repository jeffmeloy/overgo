package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

// Cohere2MTPSampledDraft: proposals plus sampler checkpoints.
type Cohere2MTPSampledDraft = Qwen35MTPSampledDraft

// DraftCohere2MTPSampled: transactional sampled proposals.
func (r *Runner) DraftCohere2MTPSampled(
	ctx context.Context,
	session *Cohere2MTPSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *Cohere2MTPSampledDraft, err error) {
	if r == nil || session == nil || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: Cohere2-MoE MTP sampled draft inputs are invalid")
	}
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 ||
		math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Cohere2-MoE MTP sampled draft limits are invalid")
	}
	initialState, err := sampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Cohere2-MoE MTP draft sampler: %w", err)
	}
	defer func() {
		restoreErr := sampler.LoadState(initialState)
		if err == nil && restoreErr != nil {
			draft = nil
			err = fmt.Errorf("inference: restore Cohere2-MoE MTP draft sampler: %w", restoreErr)
		}
	}()
	initialToken := history[len(history)-1]
	draft = &Cohere2MTPSampledDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum),
		Distributions: make([][]sampling.TokenProbability, 0, maximum),
		History:       append([]tokenizer.TokenID(nil), history...), Base: session,
		samplerStates: [][]byte{append([]byte(nil), initialState...)},
	}
	currentToken := initialToken
	currentSession := session
	currentHistory := tokenIDsAsInts(history)
	for range maximum {
		logits, next, advanceErr := r.AdvanceCohere2MTP(ctx, currentToken, currentSession)
		if advanceErr != nil {
			return nil, advanceErr
		}
		result, sampleErr := sampler.SampleWithHistoryProbabilities(
			logits.Data, currentHistory, len(logits.Data),
		)
		if sampleErr != nil {
			return nil, sampleErr
		}
		if result.SelectedProbability < minimumProbability {
			break
		}
		state, stateErr := sampler.SaveState()
		if stateErr != nil {
			return nil, fmt.Errorf("inference: save Cohere2-MoE MTP draft checkpoint: %w", stateErr)
		}
		token := tokenizer.TokenID(result.Token)
		draft.Tokens = append(draft.Tokens, token)
		draft.Probabilities = append(draft.Probabilities, result.SelectedProbability)
		draft.Distributions = append(draft.Distributions, append([]sampling.TokenProbability(nil), result.Top...))
		draft.samplerStates = append(draft.samplerStates, state)
		currentToken, currentSession = token, next
		currentHistory = append(currentHistory, result.Token)
		if r.vocab.IsEOG(token) {
			break
		}
	}
	return draft, nil
}

// VerifyCohere2MTPSampled: ratio verification plus rollback.
func (r *Runner) VerifyCohere2MTPSampled(
	ctx context.Context,
	target *Runner,
	draft *Cohere2MTPSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *Cohere2MTPVerification, err error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil ||
		draftSampler == nil || targetSampler == nil || draftSampler == targetSampler {
		return nil, errors.New("inference: Cohere2-MoE MTP sampled verification inputs are invalid")
	}
	if len(draft.History) == 0 || draft.History[len(draft.History)-1] != draft.InitialToken ||
		len(draft.Tokens) != len(draft.Probabilities) || len(draft.Tokens) != len(draft.Distributions) ||
		len(draft.samplerStates) != len(draft.Tokens)+1 {
		return nil, errors.New("inference: Cohere2-MoE MTP sampled draft state is inconsistent")
	}
	if r.weights.Cohere2MTP != nil && r.weights.Cohere2MTP.MTPOnly {
		if err := r.validateCohere2MTPTarget(target); err != nil {
			return nil, err
		}
	} else if r != target {
		return nil, errors.New("inference: bundled Cohere2-MoE MTP verification requires its owning target runner")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: Cohere2-MoE MTP session belongs to a different target model")
	}
	draftOriginal, err := draftSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Cohere2-MoE MTP draft sampler: %w", err)
	}
	targetOriginal, err := targetSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Cohere2-MoE MTP target sampler: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = draftSampler.LoadState(draftOriginal)
			_ = targetSampler.LoadState(targetOriginal)
		}
	}()
	if err := draftSampler.LoadState(draft.samplerStates[0]); err != nil {
		return nil, fmt.Errorf("inference: restore Cohere2-MoE MTP draft base: %w", err)
	}
	mtpSession := draft.Base
	targetCache := draft.Base.TrunkCache
	currentToken := draft.InitialToken
	history := tokenIDsAsInts(draft.History)
	accepted := 0
	for {
		logits, nextSession, advanceErr := r.advanceCohere2MTPVerification(
			ctx, target, currentToken, mtpSession, targetCache,
		)
		if advanceErr != nil {
			return nil, advanceErr
		}
		mtpSession, targetCache = nextSession, nextSession.TrunkCache
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
			return nil, fmt.Errorf("inference: restore Cohere2-MoE MTP accepted draft: %w", err)
		}
		if err := draftSampler.AcceptToken(int(nextToken)); err != nil {
			return nil, fmt.Errorf("inference: commit Cohere2-MoE target token to draft sampler: %w", err)
		}
		verification = &Cohere2MTPVerification{
			Accepted: accepted, NextToken: nextToken, TargetLogits: logits, Session: mtpSession,
		}
		committed = true
		return verification, nil
	}
}
