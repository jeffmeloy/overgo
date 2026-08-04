package inference

import (
	"context"
	"errors"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Qwen35MTPSampledDraft: proposals plus exact sampler checkpoints.
type Qwen35MTPSampledDraft = sampledDraft[*Qwen35MTPSession]

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
	if !validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: Qwen3.5 MTP sampled draft limits are invalid")
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "Qwen3.5 MTP", r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Qwen35MTPSession, _ int) ([]float32, *Qwen35MTPSession, error) {
			logits, next, advanceErr := r.AdvanceQwen35MTP(ctx, token, state)
			return logits.Data, next, advanceErr
		},
	)
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
		!validVerificationSamplers(draftSampler, targetSampler) {
		return nil, errors.New("inference: Qwen3.5 MTP sampled verification inputs are invalid")
	}
	if !validSampledDraft(draft) {
		return nil, errors.New("inference: Qwen3.5 MTP sampled draft state is inconsistent")
	}
	mtpOnly := r.weights.Qwen35MTP != nil && r.weights.Qwen35MTP.MTPOnly
	if err := r.validateSingleHeadMTPVerificationTarget(
		target, draft.Base, mtpOnly, "Qwen3.5 MTP", r.validateQwen35MTPTarget,
	); err != nil {
		return nil, err
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, "Qwen3.5 MTP",
		func(token tokenizer.TokenID, state *Qwen35MTPSession) (reference.Value, *Qwen35MTPSession, error) {
			return r.advanceQwen35MTPVerification(ctx, target, token, state, state.TrunkCache)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *Qwen35MTPSession) (*Qwen35MTPVerification, error) {
			return &Qwen35MTPVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
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
	nextMTPSession.PendingHidden = lastHiddenColumn(hidden)
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
