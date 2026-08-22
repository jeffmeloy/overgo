package inference

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// PairedFeatureDraft: one masked-block proposal.
type PairedFeatureDraft = greedyDraft[*PairedFeatureSession]

// PairedFeatureVerification: accepted prefix plus target correction.
type PairedFeatureVerification = greedyVerification[*PairedFeatureSession]

// DraftPairedFeatureGreedy: bounded masked-block proposals.
func (r *Runner) DraftPairedFeatureGreedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *PairedFeatureSession,
	maximum int,
	minimumProbability float64,
) (*PairedFeatureDraft, error) {
	if r == nil || target == nil || !validPairedFeatureSession(session) ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: paired-feature draft inputs are invalid")
	}
	logits, err := r.DraftPairedFeatureBlock(ctx, target, initialToken, maximum, session.Cache)
	if err != nil {
		return nil, err
	}
	index := 0
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, target.vocab.IsEOG,
		func(_ tokenizer.TokenID, state *PairedFeatureSession) (reference.Value, *PairedFeatureSession, error) {
			index++
			row, rowErr := pairedFeatureLogitRow(logits, index, target.vocab.Len())
			return reference.Value{Data: row}, state, rowErr
		},
	)
}

// VerifyPairedFeatureGreedy: target verification and feature-cache resync.
func (r *Runner) VerifyPairedFeatureGreedy(
	ctx context.Context,
	target *Runner,
	draft *PairedFeatureDraft,
) (*PairedFeatureVerification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) || !validPairedFeatureSession(draft.Base) {
		return nil, errors.New("inference: paired-feature verification inputs are invalid")
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *PairedFeatureSession,
	) (reference.Value, *PairedFeatureSession, error) {
		return r.advancePairedFeatureVerification(ctx, target, token, state)
	})
}

func validPairedFeatureSession(session *PairedFeatureSession) bool {
	return session != nil && session.Cache != nil && session.TargetCache != nil &&
		len(session.TargetTokens) > 0 && session.Position == uint32(len(session.TargetTokens)) &&
		session.Cache.Position == session.Position &&
		effectiveCachePosition(session.TargetCache) == session.Position
}

func (r *Runner) advancePairedFeatureVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	session *PairedFeatureSession,
) (reference.Value, *PairedFeatureSession, error) {
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
	tokens := append(slices.Clone(session.TargetTokens), currentToken)
	fused, err := r.projectFeatures(ctx, features)
	if err != nil {
		return reference.Value{}, nil, err
	}
	cache, err := r.InjectPairedFeatures(ctx, fused, []uint32{session.Position}, session.Cache)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &PairedFeatureSession{
		Cache: cache, TargetCache: targetCache, TargetTokens: tokens,
		Position: session.Position + 1,
	}
	if !validPairedFeatureSession(next) {
		return reference.Value{}, nil, errors.New("inference: paired-feature verification cache resync failed")
	}
	return logits, next, nil
}

func pairedFeatureLogitRow(logits reference.Value, row int, vocabulary int) ([]float32, error) {
	width, _, valid := logits.MatrixExtents()
	if !valid || width != vocabulary || row < 0 {
		return nil, errors.New("inference: paired-feature block logits are incompatible")
	}
	view := logits.RowView(uint64(row))
	if !view.Defined() {
		return nil, errors.New("inference: paired-feature block logits are incompatible")
	}
	return view.Data, nil
}
