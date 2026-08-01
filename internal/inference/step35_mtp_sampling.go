package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Step35MTPSampledDraft: proposals plus sampler checkpoints.
type Step35MTPSampledDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Distributions [][]sampling.TokenProbability
	History       []tokenizer.TokenID
	Base          *Step35MTPSession
	samplerStates [][]byte
}

type HYV3MTPSampledDraft = Step35MTPSampledDraft

// DraftStep35MTPSampled: transactional trained-head sampling.
func (r *Runner) DraftStep35MTPSampled(
	ctx context.Context,
	session *Step35MTPSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *Step35MTPSampledDraft, err error) {
	if err := r.validateStep35MTP(); err != nil {
		return nil, err
	}
	return r.draftMultiHeadMTPSampled(ctx, session, sampler, history, maximum, minimumProbability)
}

// DraftHYV3MTPSampled: transactional HY-V3 head sampling.
func (r *Runner) DraftHYV3MTPSampled(
	ctx context.Context,
	session *HYV3MTPSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (*HYV3MTPSampledDraft, error) {
	if err := r.validateHYV3MTP(); err != nil {
		return nil, err
	}
	return r.draftMultiHeadMTPSampled(ctx, session, sampler, history, maximum, minimumProbability)
}

func (r *Runner) draftMultiHeadMTPSampled(
	ctx context.Context,
	session *Step35MTPSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *Step35MTPSampledDraft, err error) {
	if r == nil || session == nil || len(session.DraftTokens) != 0 || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: Step3.5 MTP sampled draft inputs are invalid")
	}
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Step3.5 MTP sampled draft limits are invalid")
	}
	maximum = min(maximum, len(r.multiHeadMTPWeights()))
	initialState, err := sampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Step3.5 MTP draft sampler: %w", err)
	}
	defer func() {
		restoreErr := sampler.LoadState(initialState)
		if err == nil && restoreErr != nil {
			draft = nil
			err = fmt.Errorf("inference: restore Step3.5 MTP draft sampler: %w", restoreErr)
		}
	}()
	initialToken := history[len(history)-1]
	draft = &Step35MTPSampledDraft{
		InitialToken:  initialToken,
		Tokens:        make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum),
		Distributions: make([][]sampling.TokenProbability, 0, maximum),
		History:       append([]tokenizer.TokenID(nil), history...), Base: session,
		samplerStates: [][]byte{append([]byte(nil), initialState...)},
	}
	currentToken := initialToken
	currentSession := session
	currentHistory := tokenIDsAsInts(history)
	for range maximum {
		logits, next, advanceErr := r.advanceMultiHeadMTP(ctx, currentToken, currentSession)
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
			return nil, fmt.Errorf("inference: save Step3.5 MTP draft checkpoint: %w", stateErr)
		}
		token := tokenizer.TokenID(result.Token)
		draft.Tokens = append(draft.Tokens, token)
		draft.Probabilities = append(draft.Probabilities, result.SelectedProbability)
		draft.Distributions = append(
			draft.Distributions, append([]sampling.TokenProbability(nil), result.Top...),
		)
		draft.samplerStates = append(draft.samplerStates, state)
		currentToken, currentSession = token, next
		currentHistory = append(currentHistory, result.Token)
		if r.vocab.IsEOG(token) {
			break
		}
	}
	return draft, nil
}

// VerifyStep35MTPSampled: ratio verification plus cache resync.
func (r *Runner) VerifyStep35MTPSampled(
	ctx context.Context,
	target *Runner,
	draft *Step35MTPSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *Step35MTPVerification, err error) {
	if err := r.validateStep35MTP(); err != nil {
		return nil, err
	}
	return r.verifyMultiHeadMTPSampled(ctx, target, draft, draftSampler, targetSampler)
}

// VerifyHYV3MTPSampled: sampled HY-V3 target verification.
func (r *Runner) VerifyHYV3MTPSampled(
	ctx context.Context,
	target *Runner,
	draft *HYV3MTPSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (*HYV3MTPVerification, error) {
	if err := r.validateHYV3MTP(); err != nil {
		return nil, err
	}
	return r.verifyMultiHeadMTPSampled(ctx, target, draft, draftSampler, targetSampler)
}

func (r *Runner) verifyMultiHeadMTPSampled(
	ctx context.Context,
	target *Runner,
	draft *Step35MTPSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *Step35MTPVerification, err error) {
	if r == nil || target == nil || r != target || draft == nil || draft.Base == nil ||
		draftSampler == nil || targetSampler == nil || draftSampler == targetSampler {
		return nil, errors.New("inference: Step3.5 MTP sampled verification inputs are invalid")
	}
	if len(draft.History) == 0 || draft.History[len(draft.History)-1] != draft.InitialToken ||
		len(draft.Tokens) != len(draft.Probabilities) || len(draft.Tokens) != len(draft.Distributions) ||
		len(draft.samplerStates) != len(draft.Tokens)+1 || len(draft.Tokens) > len(r.multiHeadMTPWeights()) {
		return nil, errors.New("inference: Step3.5 MTP sampled draft state is inconsistent")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: Step3.5 MTP session belongs to a different target model")
	}
	draftOriginal, err := draftSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Step3.5 MTP draft sampler: %w", err)
	}
	targetOriginal, err := targetSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Step3.5 MTP target sampler: %w", err)
	}
	committed := false
	defer func() {
		if committed {
			return
		}
		_ = draftSampler.LoadState(draftOriginal)
		_ = targetSampler.LoadState(targetOriginal)
	}()
	if err := draftSampler.LoadState(draft.samplerStates[0]); err != nil {
		return nil, fmt.Errorf("inference: restore Step3.5 MTP draft base: %w", err)
	}
	targetCache := draft.Base.TrunkCache
	currentToken := draft.InitialToken
	history := tokenIDsAsInts(draft.History)
	processedTokens := make([]tokenizer.TokenID, 0, len(draft.Tokens)+1)
	processedHidden := make([]reference.Value, 0, len(draft.Tokens)+1)
	accepted := 0
	for {
		logits, hidden, nextTargetCache, advanceErr := target.multiHeadMTPTargetAdvance(
			ctx, currentToken, targetCache,
		)
		if advanceErr != nil {
			return nil, advanceErr
		}
		processedTokens = append(processedTokens, currentToken)
		processedHidden = append(processedHidden, hidden)
		targetCache = nextTargetCache
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
			return nil, fmt.Errorf("inference: restore Step3.5 MTP accepted draft: %w", err)
		}
		if err := draftSampler.AcceptToken(int(nextToken)); err != nil {
			return nil, fmt.Errorf("inference: commit Step3.5 MTP target token to draft sampler: %w", err)
		}
		session, syncErr := r.resyncStep35MTPSession(
			ctx, draft.Base, targetCache, processedTokens, processedHidden,
		)
		if syncErr != nil {
			return nil, syncErr
		}
		verification = &Step35MTPVerification{
			Accepted: accepted, NextToken: nextToken,
			TargetLogits: logits, Session: session,
		}
		committed = true
		return verification, nil
	}
}
