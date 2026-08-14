package inference

import (
	"context"
	"errors"

	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// PairedProjectionDraft: bounded greedy proposals.
type PairedProjectionDraft = greedyDraft[*PairedProjectionSession]

// PairedProjectionVerification: accepted prefix plus correction.
type PairedProjectionVerification = greedyVerification[*PairedProjectionSession]

// DraftPairedProjectionGreedy: bounded confident proposals.
func (r *Runner) DraftPairedProjectionGreedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *PairedProjectionSession,
	maximum int,
	minimumProbability float64,
) (*PairedProjectionDraft, error) {
	if r == nil || target == nil || session == nil ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: paired projection draft inputs are invalid")
	}
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, target.vocab.IsEOG,
		func(token tokenizer.TokenID, state *PairedProjectionSession) (reference.Value, *PairedProjectionSession, error) {
			return r.AdvancePairedProjection(ctx, target, token, state)
		},
	)
}

// VerifyPairedProjectionGreedy: target verification plus hidden resync.
func (r *Runner) VerifyPairedProjectionGreedy(
	ctx context.Context,
	target *Runner,
	draft *PairedProjectionDraft,
) (*PairedProjectionVerification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) ||
		draft.Base == nil || draft.Base.TargetCache == nil {
		return nil, errors.New("inference: paired projection verification inputs are invalid")
	}
	if err := r.validatePairedProjectionTarget(target); err != nil {
		return nil, err
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *PairedProjectionSession,
	) (reference.Value, *PairedProjectionSession, error) {
		return r.advancePairedProjectionVerification(
			ctx, target, token, state, state.TargetCache,
		)
	})
}
