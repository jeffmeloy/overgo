package inference

import (
	"context"
	"errors"

	"overgo/internal/sampling"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// PairedProjectionSampledDraft: proposals plus sampler checkpoints.
type PairedProjectionSampledDraft = sampledDraft[*PairedProjectionSession]

// DraftPairedProjectionSampled: transactional sampled proposals.
func (r *Runner) DraftPairedProjectionSampled(
	ctx context.Context,
	target *Runner,
	session *PairedProjectionSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *PairedProjectionSampledDraft, err error) {
	if r == nil || target == nil || session == nil || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: paired projection sampled draft inputs are invalid")
	}
	if !validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: paired projection sampled draft limits are invalid")
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "paired projection", target.vocab.IsEOG,
		func(token tokenizer.TokenID, state *PairedProjectionSession, _ int) ([]float32, *PairedProjectionSession, error) {
			logits, next, advanceErr := r.AdvancePairedProjection(ctx, target, token, state)
			return logits.Data, next, advanceErr
		},
	)
}

// VerifyPairedProjectionSampled: ratio verification plus rollback.
func (r *Runner) VerifyPairedProjectionSampled(
	ctx context.Context,
	target *Runner,
	draft *PairedProjectionSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *PairedProjectionVerification, err error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil || draft.Base.TargetCache == nil ||
		!validVerificationSamplers(draftSampler, targetSampler) {
		return nil, errors.New("inference: paired projection sampled verification inputs are invalid")
	}
	if !validSampledDraft(draft) {
		return nil, errors.New("inference: paired projection sampled draft state is inconsistent")
	}
	if err := r.validatePairedProjectionTarget(target); err != nil {
		return nil, err
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, "paired projection",
		func(token tokenizer.TokenID, state *PairedProjectionSession) (reference.Value, *PairedProjectionSession, error) {
			return r.advancePairedProjectionVerification(ctx, target, token, state, state.TargetCache)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *PairedProjectionSession) (*PairedProjectionVerification, error) {
			return &PairedProjectionVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
}

func (r *Runner) advancePairedProjectionVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	assistantSession *PairedProjectionSession,
	targetCache *KVCache,
) (reference.Value, *PairedProjectionSession, error) {
	return advanceTargetVerification(
		ctx, target, currentToken, targetCache,
		func() (*PairedProjectionSession, error) {
			_, next, err := r.AdvancePairedProjection(
				ctx, target, currentToken, assistantSession,
			)
			return next, err
		},
		func(next *PairedProjectionSession, hidden reference.Value, cache *KVCache) {
			next.TargetCache = cache
			next.PendingHidden = hidden.LastRowView().Clone()
			next.Position = effectiveCachePosition(cache)
		},
	)
}
