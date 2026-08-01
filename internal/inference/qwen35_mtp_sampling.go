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

// Qwen35MTPSampledDraft: proposals plus exact sampler checkpoints.
type Qwen35MTPSampledDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Distributions [][]sampling.TokenProbability
	History       []tokenizer.TokenID
	Base          *Qwen35MTPSession
	samplerStates [][]byte
}

// DraftQwen35MTPSampled: transactional sampled proposal chain.
func (r *Runner) DraftQwen35MTPSampled(
	ctx context.Context,
	session *Qwen35MTPSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *Qwen35MTPSampledDraft, err error) {
	if r == nil || session == nil || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: Qwen3.5 MTP sampled draft inputs are invalid")
	}
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 ||
		math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Qwen3.5 MTP sampled draft limits are invalid")
	}
	initialState, err := sampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Qwen3.5 MTP draft sampler: %w", err)
	}
	defer func() {
		restoreErr := sampler.LoadState(initialState)
		if err == nil && restoreErr != nil {
			draft = nil
			err = fmt.Errorf("inference: restore Qwen3.5 MTP draft sampler: %w", restoreErr)
		}
	}()
	initialToken := history[len(history)-1]
	draft = &Qwen35MTPSampledDraft{
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
		logits, next, advanceErr := r.AdvanceQwen35MTP(ctx, currentToken, currentSession)
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
			return nil, fmt.Errorf("inference: save Qwen3.5 MTP draft checkpoint: %w", stateErr)
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

// VerifyQwen35MTPSampled: ratio verification plus sampler rollback.
func (r *Runner) VerifyQwen35MTPSampled(
	ctx context.Context,
	target *Runner,
	draft *Qwen35MTPSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *Qwen35MTPVerification, err error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil ||
		draftSampler == nil || targetSampler == nil || draftSampler == targetSampler {
		return nil, errors.New("inference: Qwen3.5 MTP sampled verification inputs are invalid")
	}
	if len(draft.History) == 0 || draft.History[len(draft.History)-1] != draft.InitialToken ||
		len(draft.Tokens) != len(draft.Probabilities) ||
		len(draft.Tokens) != len(draft.Distributions) ||
		len(draft.samplerStates) != len(draft.Tokens)+1 {
		return nil, errors.New("inference: Qwen3.5 MTP sampled draft state is inconsistent")
	}
	if r.weights.Qwen35MTP != nil && r.weights.Qwen35MTP.MTPOnly {
		if err := r.validateQwen35MTPTarget(target); err != nil {
			return nil, err
		}
	} else if r != target {
		return nil, errors.New("inference: bundled Qwen3.5 MTP verification requires its owning target runner")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: Qwen3.5 MTP session belongs to a different target model")
	}
	draftOriginal, err := draftSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Qwen3.5 MTP draft sampler: %w", err)
	}
	targetOriginal, err := targetSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save Qwen3.5 MTP target sampler: %w", err)
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
		return nil, fmt.Errorf("inference: restore Qwen3.5 MTP draft base: %w", err)
	}
	mtpSession := draft.Base
	targetCache := draft.Base.TrunkCache
	currentToken := draft.InitialToken
	history := tokenIDsAsInts(draft.History)
	accepted := 0
	for {
		logits, nextSession, advanceErr := r.advanceQwen35MTPVerification(
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
			return nil, fmt.Errorf("inference: restore Qwen3.5 MTP accepted draft: %w", err)
		}
		if err := draftSampler.AcceptToken(int(nextToken)); err != nil {
			return nil, fmt.Errorf("inference: commit Qwen3.5 MTP target token to draft sampler: %w", err)
		}
		verification = &Qwen35MTPVerification{
			Accepted: accepted, NextToken: nextToken,
			TargetLogits: logits, Session: mtpSession,
		}
		committed = true
		return verification, nil
	}
}

func (r *Runner) advanceQwen35MTPVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	mtpSession *Qwen35MTPSession,
	targetCache *KVCache,
) (reference.Value, *Qwen35MTPSession, error) {
	hidden, nextTargetCache, err := target.ForwardCached(
		ctx, []tokenizer.TokenID{currentToken}, targetCache,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, err := target.qwen35MTPProjectLogits(ctx, hidden)
	if err != nil {
		return reference.Value{}, nil, err
	}
	_, nextMTPSession, err := r.AdvanceQwen35MTP(ctx, currentToken, mtpSession)
	if err != nil {
		return reference.Value{}, nil, err
	}
	width := int(hidden.Shape.Dims[0])
	nextMTPSession.PendingHidden = reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), hidden.Data[len(hidden.Data)-width:]...),
	}
	nextMTPSession.TrunkCache = nextTargetCache
	return logits, nextMTPSession, nil
}

func tokenIDsAsInts(ids []tokenizer.TokenID) []int {
	result := make([]int, len(ids))
	for index, id := range ids {
		result[index] = int(id)
	}
	return result
}
