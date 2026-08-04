package inference

import (
	"context"
	"errors"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Gemma4AssistantSampledDraft: proposals plus sampler checkpoints.
type Gemma4AssistantSampledDraft = sampledDraft[*Gemma4AssistantSession]

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
	if !validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: Gemma 4 assistant sampled draft limits are invalid")
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "Gemma 4 assistant", target.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Gemma4AssistantSession, _ int) ([]float32, *Gemma4AssistantSession, error) {
			logits, next, advanceErr := r.AdvanceGemma4Assistant(ctx, target, token, state)
			return logits.Data, next, advanceErr
		},
	)
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
		!validVerificationSamplers(draftSampler, targetSampler) {
		return nil, errors.New("inference: Gemma 4 assistant sampled verification inputs are invalid")
	}
	if !validSampledDraft(draft) {
		return nil, errors.New("inference: Gemma 4 assistant sampled draft state is inconsistent")
	}
	if err := r.validateGemma4AssistantTarget(target); err != nil {
		return nil, err
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, "Gemma 4 assistant",
		func(token tokenizer.TokenID, state *Gemma4AssistantSession) (reference.Value, *Gemma4AssistantSession, error) {
			return r.advanceGemma4AssistantVerification(ctx, target, token, state, state.TargetCache)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *Gemma4AssistantSession) (*Gemma4AssistantVerification, error) {
			return &Gemma4AssistantVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
}

func (r *Runner) advanceGemma4AssistantVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	assistantSession *Gemma4AssistantSession,
	targetCache *KVCache,
) (reference.Value, *Gemma4AssistantSession, error) {
	return advanceTargetVerification(
		ctx, target, currentToken, targetCache,
		func() (*Gemma4AssistantSession, error) {
			_, next, err := r.AdvanceGemma4Assistant(
				ctx, target, currentToken, assistantSession,
			)
			return next, err
		},
		func(next *Gemma4AssistantSession, hidden reference.Value, cache *KVCache) {
			next.TargetCache = cache
			next.PendingHidden = lastHiddenColumn(hidden)
			next.Position = effectiveCachePosition(cache)
		},
	)
}
