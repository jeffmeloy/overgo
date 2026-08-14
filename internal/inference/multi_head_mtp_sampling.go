package inference

import (
	"context"
	"errors"

	"overgo/internal/sampling"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// MultiHeadMTPSampledDraft: proposals plus sampler checkpoints.
type MultiHeadMTPSampledDraft = sampledDraft[*MultiHeadMTPSession]

// DraftMultiHeadMTPSampled performs transactional trained-head sampling.
func (r *Runner) DraftMultiHeadMTPSampled(
	ctx context.Context,
	session *MultiHeadMTPSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *MultiHeadMTPSampledDraft, err error) {
	plan, err := r.multiHeadMTP()
	if err != nil {
		return nil, err
	}
	if r == nil || session == nil || len(session.DraftTokens) != 0 || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: multi-head MTP sampled draft inputs are invalid")
	}
	if !validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: multi-head MTP sampled draft limits are invalid")
	}
	maximum = min(maximum, int(plan.Heads))
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, mtpLabel, r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *MultiHeadMTPSession, _ int) ([]float32, *MultiHeadMTPSession, error) {
			logits, next, advanceErr := r.AdvanceMultiHeadMTP(ctx, token, state)
			return logits.Data, next, advanceErr
		},
	)
}

// VerifyMultiHeadMTPSampled performs ratio verification plus cache resync.
func (r *Runner) VerifyMultiHeadMTPSampled(
	ctx context.Context,
	target *Runner,
	draft *MultiHeadMTPSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *MultiHeadMTPVerification, err error) {
	plan, err := r.multiHeadMTP()
	if err != nil {
		return nil, err
	}
	if r == nil || target == nil || r != target || draft == nil || draft.Base == nil ||
		!validVerificationSamplers(draftSampler, targetSampler) {
		return nil, errors.New("inference: multi-head MTP sampled verification inputs are invalid")
	}
	if !validSampledDraft(draft) || len(draft.Tokens) > int(plan.Heads) {
		return nil, errors.New("inference: multi-head MTP sampled draft state is inconsistent")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: multi-head MTP session belongs to a different target model")
	}
	state := multiHeadVerificationState{
		cache:  draft.Base.TrunkCache,
		tokens: make([]tokenizer.TokenID, 0, len(draft.Tokens)+1),
		hidden: make([]reference.Value, 0, len(draft.Tokens)+1),
	}
	return verifySampled(
		draft, state, draftSampler, targetSampler, mtpLabel,
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
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state multiHeadVerificationState) (*MultiHeadMTPVerification, error) {
			session, syncErr := r.resyncMultiHeadMTPSession(ctx, draft.Base, state.cache, state.tokens, state.hidden)
			if syncErr != nil {
				return nil, syncErr
			}
			return &MultiHeadMTPVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: session}, nil
		},
	)
}

type multiHeadVerificationState struct {
	cache  *KVCache
	tokens []tokenizer.TokenID
	hidden []reference.Value
}
