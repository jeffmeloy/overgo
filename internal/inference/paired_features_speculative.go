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
	index := tensor.FirstOffset
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
		checked.Nonzero(len(session.TargetTokens)) && session.Position == uint32(len(session.TargetTokens)) &&
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
		Position: session.Position + uint32(tensor.SingletonExtent),
	}
	if !validPairedFeatureSession(next) {
		return reference.Value{}, nil, errors.New("inference: paired-feature verification cache resync failed")
	}
	return logits, next, nil
}

func pairedFeatureLogitRow(logits reference.Value, row int, vocabulary int) ([]float32, error) {
	rowIndex, validIndex := checked.Uint64(int64(row))
	rows, validMatrix := tensor.MatrixRows(logits.Shape, uint64(vocabulary))
	if !checked.PositiveInts(vocabulary) || !validIndex || !validMatrix || !checked.Less64(rowIndex, rows) {
		return nil, errors.New("inference: paired-feature block logits are incompatible")
	}
	selected, err := reference.SelectRows(logits, rowIndex, tensor.SingletonExtent)
	if err != nil {
		return nil, errors.New("inference: paired-feature block logits are incompatible")
	}
	return selected.Data, nil
}
