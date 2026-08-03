package inference

import (
	"context"
	"errors"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// NextNMTPDraft: greedy dense-tail proposals.
type NextNMTPDraft = Qwen35MTPDraft

// NextNMTPVerification: accepted prefix plus correction.
type NextNMTPVerification = Qwen35MTPVerification

// DraftNextNMTPGreedy: bounded confident proposals.
func (r *Runner) DraftNextNMTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *NextNMTPSession,
	maximum int,
	minimumProbability float64,
) (*NextNMTPDraft, error) {
	if !validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: NextN MTP draft limits are invalid")
	}
	if session == nil {
		return nil, errors.New("inference: NextN MTP draft session is nil")
	}
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *NextNMTPSession) (reference.Value, *NextNMTPSession, error) {
			return r.AdvanceNextNMTP(ctx, token, state)
		},
	)
}

// VerifyNextNMTPGreedy: target check and resync.
func (r *Runner) VerifyNextNMTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *NextNMTPDraft,
) (*NextNMTPVerification, error) {
	if r == nil || target == nil || r != target || !validGreedyDraft(draft) || draft.Base == nil {
		return nil, errors.New("inference: bundled NextN MTP verification inputs are invalid")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: NextN MTP session belongs to a different target model")
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *NextNMTPSession,
	) (reference.Value, *NextNMTPSession, error) {
		hidden, cache, err := target.ForwardCached(ctx, []tokenizer.TokenID{token}, state.TrunkCache)
		if err != nil {
			return reference.Value{}, nil, err
		}
		logits, err := target.projectHiddenLogits(ctx, hidden)
		if err != nil {
			return reference.Value{}, nil, err
		}
		_, next, err := r.AdvanceNextNMTP(ctx, token, state)
		if err != nil {
			return reference.Value{}, nil, err
		}
		next.PendingHidden = lastHiddenColumn(hidden)
		next.TrunkCache = cache
		return logits, next, nil
	})
}
