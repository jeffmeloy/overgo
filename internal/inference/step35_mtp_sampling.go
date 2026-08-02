package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Step35MTPSampledDraft: proposals plus sampler checkpoints.
type Step35MTPSampledDraft = sampledDraft[*Step35MTPSession]

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
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "Step3.5 MTP", r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Step35MTPSession, _ int) ([]float32, *Step35MTPSession, error) {
			logits, next, advanceErr := r.advanceMultiHeadMTP(ctx, token, state)
			return logits.Data, next, advanceErr
		},
	)
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
	if !validSampledDraft(draft) || len(draft.Tokens) > len(r.multiHeadMTPWeights()) {
		return nil, errors.New("inference: Step3.5 MTP sampled draft state is inconsistent")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: Step3.5 MTP session belongs to a different target model")
	}
	state := multiHeadVerificationState{
		cache:  draft.Base.TrunkCache,
		tokens: make([]tokenizer.TokenID, 0, len(draft.Tokens)+1),
		hidden: make([]reference.Value, 0, len(draft.Tokens)+1),
	}
	return verifySampled(
		draft, state, draftSampler, targetSampler, "Step3.5 MTP",
		func(token tokenizer.TokenID, state multiHeadVerificationState) (reference.Value, multiHeadVerificationState, error) {
			logits, hidden, cache, advanceErr := target.multiHeadMTPTargetAdvance(ctx, token, state.cache)
			if advanceErr != nil {
				return reference.Value{}, state, advanceErr
			}
			state.cache = cache
			state.tokens = append(state.tokens, token)
			state.hidden = append(state.hidden, hidden)
			return logits, state, nil
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state multiHeadVerificationState) (*Step35MTPVerification, error) {
			session, syncErr := r.resyncStep35MTPSession(ctx, draft.Base, state.cache, state.tokens, state.hidden)
			if syncErr != nil {
				return nil, syncErr
			}
			return &Step35MTPVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: session}, nil
		},
	)
}

type multiHeadVerificationState struct {
	cache  *KVCache
	tokens []tokenizer.TokenID
	hidden []reference.Value
}
