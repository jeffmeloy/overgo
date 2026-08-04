package inference

import (
	"context"
	"errors"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Cohere2MTPDraft: greedy single-block proposals.
type Cohere2MTPDraft = Qwen35MTPDraft

// Cohere2MTPVerification: accepted prefix plus correction.
type Cohere2MTPVerification = Qwen35MTPVerification

// DraftCohere2MTPGreedy: bounded confident proposals.
func (r *Runner) DraftCohere2MTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *Cohere2MTPSession,
	maximum int,
	minimumProbability float64,
) (*Cohere2MTPDraft, error) {
	if !validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: Cohere2-MoE MTP draft limits are invalid")
	}
	if session == nil {
		return nil, errors.New("inference: Cohere2-MoE MTP draft session is nil")
	}
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Cohere2MTPSession) (reference.Value, *Cohere2MTPSession, error) {
			return r.AdvanceCohere2MTP(ctx, token, state)
		},
	)
}

// VerifyCohere2MTPGreedy: target check plus hidden resync.
func (r *Runner) VerifyCohere2MTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *Cohere2MTPDraft,
) (*Cohere2MTPVerification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) || draft.Base == nil {
		return nil, errors.New("inference: Cohere2-MoE MTP verification inputs are invalid")
	}
	mtpOnly := r.weights.Cohere2MTP != nil && r.weights.Cohere2MTP.MTPOnly
	if err := r.validateSingleHeadMTPVerificationTarget(
		target, draft.Base, mtpOnly, "Cohere2-MoE MTP", r.validateCohere2MTPTarget,
	); err != nil {
		return nil, err
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *Cohere2MTPSession,
	) (reference.Value, *Cohere2MTPSession, error) {
		return r.advanceCohere2MTPVerification(ctx, target, token, state, state.TrunkCache)
	})
}

func (r *Runner) advanceCohere2MTPVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	mtpSession *Cohere2MTPSession,
	targetCache *KVCache,
) (reference.Value, *Cohere2MTPSession, error) {
	return r.advanceSingleHeadMTPVerification(
		ctx, target, currentToken, mtpSession, targetCache,
		func(token tokenizer.TokenID, state *Qwen35MTPSession) (reference.Value, *Qwen35MTPSession, error) {
			return r.AdvanceCohere2MTP(ctx, token, state)
		},
	)
}
