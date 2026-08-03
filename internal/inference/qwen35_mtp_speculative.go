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
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *Qwen35MTPSession,
	) (reference.Value, *Qwen35MTPSession, error) {
		hidden, cache, err := target.ForwardCached(ctx, []tokenizer.TokenID{token}, state.TrunkCache)
		if err != nil {
			return reference.Value{}, nil, err
		}
		logits, err := target.projectHiddenLogits(ctx, hidden)
		if err != nil {
			return reference.Value{}, nil, err
		}
		_, next, err := r.AdvanceQwen35MTP(ctx, token, state)
		if err != nil {
			return reference.Value{}, nil, err
		}
		next.PendingHidden = lastHiddenColumn(hidden)
		next.TrunkCache = cache
		return logits, next, nil
	})
}

func (r *Runner) qwen35MTPProjectLogits(
	ctx context.Context,
	hidden reference.Value,
) (reference.Value, error) {
	return r.projectHiddenLogits(ctx, hidden)
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
