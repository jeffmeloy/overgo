package inference

import (
	"context"
	"errors"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Qwen35MTPDraft: greedy proposals from one target-selected token.
type Qwen35MTPDraft = greedyDraft[*Qwen35MTPSession]

// Qwen35MTPVerification: accepted prefix plus target correction.
type Qwen35MTPVerification = greedyVerification[*Qwen35MTPSession]

// DraftQwen35MTPGreedy: bounded high-confidence proposal chain.
func (r *Runner) DraftQwen35MTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *Qwen35MTPSession,
	maximum int,
	minimumProbability float64,
) (*Qwen35MTPDraft, error) {
	if !validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: Qwen3.5 MTP draft limits are invalid")
	}
	if session == nil {
		return nil, errors.New("inference: Qwen3.5 MTP draft session is nil")
	}
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Qwen35MTPSession) (reference.Value, *Qwen35MTPSession, error) {
			return r.AdvanceQwen35MTP(ctx, token, state)
		},
	)
}

// VerifyQwen35MTPGreedy: target check, rollback, and hidden-state resync.
func (r *Runner) VerifyQwen35MTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *Qwen35MTPDraft,
) (*Qwen35MTPVerification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) || draft.Base == nil {
		return nil, errors.New("inference: Qwen3.5 MTP verification inputs are invalid")
	}
	mtpOnly := r.weights.Qwen35MTP != nil && r.weights.Qwen35MTP.MTPOnly
	if err := r.validateSingleHeadMTPVerificationTarget(
		target, draft.Base, mtpOnly, "Qwen3.5 MTP", r.validateQwen35MTPTarget,
	); err != nil {
		return nil, err
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *Qwen35MTPSession,
	) (reference.Value, *Qwen35MTPSession, error) {
		return r.advanceQwen35MTPVerification(ctx, target, token, state, state.TrunkCache)
	})
}

func (r *Runner) projectHiddenLogits(
	ctx context.Context,
	hidden reference.Value,
) (reference.Value, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: target runner is unavailable")
	}
	return r.projectAllLogits(ctx, hidden)
}
