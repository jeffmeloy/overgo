package inference

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// Eagle3Draft: bounded greedy proposals.
type Eagle3Draft = greedyDraft[*Eagle3Session]

// Eagle3Verification: accepted prefix plus correction.
type Eagle3Verification = greedyVerification[*Eagle3Session]

// DraftEagle3Greedy: bounded confident proposals.
func (r *Runner) DraftEagle3Greedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *Eagle3Session,
	maximum int,
	minimumProbability float64,
) (*Eagle3Draft, error) {
	if r == nil || target == nil || session == nil ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: Eagle3 draft inputs are invalid")
	}
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, target.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Eagle3Session) (reference.Value, *Eagle3Session, error) {
			return r.AdvanceEagle3(ctx, target, token, state)
		},
	)
}

// VerifyEagle3Greedy: target verification and feature resync.
func (r *Runner) VerifyEagle3Greedy(
	ctx context.Context,
	target *Runner,
	draft *Eagle3Draft,
) (*Eagle3Verification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) ||
		!validEagle3CoordinatorSession(draft.Base) {
		return nil, errors.New("inference: Eagle3 verification inputs are invalid")
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *Eagle3Session,
	) (reference.Value, *Eagle3Session, error) {
		return r.advanceEagle3Verification(ctx, target, token, state)
	})
}

func validEagle3CoordinatorSession(session *Eagle3Session) bool {
	return session != nil && session.TargetCache != nil && len(session.TargetTokens) > 0 &&
		effectiveCachePosition(session.TargetCache) == uint32(len(session.TargetTokens)) &&
		session.Position+1 == uint32(len(session.TargetTokens))
}

func (r *Runner) advanceEagle3Verification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	session *Eagle3Session,
) (reference.Value, *Eagle3Session, error) {
	hidden, targetCache, features, err := target.ForwardCachedExtractLayerInputs(
		ctx, []tokenizer.TokenID{currentToken}, session.TargetCache, r.spec.TargetLayers,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, err := target.projectHiddenLogits(ctx, hidden)
	if err != nil {
		return reference.Value{}, nil, err
	}
	_, next, err := r.AdvanceEagle3(ctx, target, currentToken, session)
	if err != nil {
		return reference.Value{}, nil, err
	}
	tokens := append(slices.Clone(session.TargetTokens), currentToken)
	fused, err := r.FuseEagle3Features(ctx, features)
	if err != nil {
		return reference.Value{}, nil, err
	}
	width := int(fused.Shape.Dims[0])
	next.TargetCache = targetCache
	next.TargetTokens = tokens
	next.PendingFeature = reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  slices.Clone(fused.Data[len(fused.Data)-width:]),
	}
	return logits, next, nil
}
