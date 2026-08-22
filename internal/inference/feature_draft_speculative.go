package inference

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// FeatureDraft: bounded greedy proposals.
type FeatureDraft = greedyDraft[*FeatureDraftSession]

// FeatureDraftVerification: accepted prefix plus correction.
type FeatureDraftVerification = greedyVerification[*FeatureDraftSession]

// DraftFeaturesGreedy: bounded confident proposals.
func (r *Runner) DraftFeaturesGreedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *FeatureDraftSession,
	maximum int,
	minimumProbability float64,
) (*FeatureDraft, error) {
	if r == nil || target == nil || !validFeatureDraftSession(session) ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: feature-draft inputs are invalid")
	}
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, target.vocab.IsEOG,
		func(token tokenizer.TokenID, state *FeatureDraftSession) (reference.Value, *FeatureDraftSession, error) {
			return r.AdvanceFeatureDraft(ctx, target, token, state)
		},
	)
}

// VerifyFeatureDraftGreedy: target verification and feature resync.
func (r *Runner) VerifyFeatureDraftGreedy(
	ctx context.Context,
	target *Runner,
	draft *FeatureDraft,
) (*FeatureDraftVerification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) ||
		!validFeatureDraftSession(draft.Base) {
		return nil, errors.New("inference: feature-draft verification inputs are invalid")
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *FeatureDraftSession,
	) (reference.Value, *FeatureDraftSession, error) {
		return r.advanceFeatureDraftVerification(ctx, target, token, state)
	})
}

func validFeatureDraftSession(session *FeatureDraftSession) bool {
	if session == nil {
		return false
	}
	nextPosition := session.Position
	nextPosition++
	return session.TargetCache != nil && checked.Nonzero(len(session.TargetTokens)) &&
		effectiveCachePosition(session.TargetCache) == uint32(len(session.TargetTokens)) &&
		nextPosition == uint32(len(session.TargetTokens))
}

func (r *Runner) advanceFeatureDraftVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	session *FeatureDraftSession,
) (reference.Value, *FeatureDraftSession, error) {
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
	_, next, err := r.AdvanceFeatureDraft(ctx, target, currentToken, session)
	if err != nil {
		return reference.Value{}, nil, err
	}
	tokens := append(slices.Clone(session.TargetTokens), currentToken)
	fused, err := r.projectFeatures(ctx, features)
	if err != nil {
		return reference.Value{}, nil, err
	}
	pending, err := reference.FinalRows(fused, tensor.SingletonExtent)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next.TargetCache = targetCache
	next.TargetTokens = tokens
	next.PendingFeature = pending
	return logits, next, nil
}
