package inference

import (
	"context"
	"errors"

	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type MTPDraft = greedyDraft[*MTPSession]
type MTPVerification = greedyVerification[*MTPSession]

func (r *Runner) DraftMTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *MTPSession,
	maximum int,
	minimumProbability float64,
) (*MTPDraft, error) {
	if !validSampledLimits(maximum, minimumProbability) || session == nil {
		return nil, errors.New("inference: MTP draft inputs are invalid")
	}
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *MTPSession) (reference.Value, *MTPSession, error) {
			return r.AdvanceMTP(ctx, token, state)
		},
	)
}

func (r *Runner) VerifyMTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *MTPDraft,
) (*MTPVerification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) || draft.Base == nil {
		return nil, errors.New("inference: MTP verification inputs are invalid")
	}
	if err := r.validateMTPVerificationTarget(target, draft.Base); err != nil {
		return nil, err
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *MTPSession,
	) (reference.Value, *MTPSession, error) {
		return r.advanceMTPVerification(ctx, target, token, state, state.TrunkCache)
	})
}

func (r *Runner) advanceMTPVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	mtpSession *MTPSession,
	targetCache *KVCache,
) (reference.Value, *MTPSession, error) {
	return r.advanceSingleHeadMTPVerification(
		ctx, target, currentToken, mtpSession, targetCache,
		func(token tokenizer.TokenID, state *MTPSession) (reference.Value, *MTPSession, error) {
			return r.AdvanceMTP(ctx, token, state)
		},
	)
}

func (r *Runner) projectHiddenLogits(ctx context.Context, hidden reference.Value) (reference.Value, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: target runner is unavailable")
	}
	return r.projectAllLogits(ctx, hidden)
}
