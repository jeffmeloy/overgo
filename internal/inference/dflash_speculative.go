package inference

import (
	"context"
	"errors"
	"slices"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// DFlashDraft: one masked-block proposal.
type DFlashDraft = greedyDraft[*DFlashSession]

// DFlashVerification: accepted prefix plus target correction.
type DFlashVerification = greedyVerification[*DFlashSession]

// DraftDFlashGreedy: bounded masked-block proposals.
func (r *Runner) DraftDFlashGreedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *DFlashSession,
	maximum int,
	minimumProbability float64,
) (*DFlashDraft, error) {
	if r == nil || target == nil || !validDFlashSession(session) ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: DFlash draft inputs are invalid")
	}
	logits, err := r.DraftDFlashBlock(ctx, target, initialToken, maximum, session.Cache)
	if err != nil {
		return nil, err
	}
	draft := &DFlashDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum), Base: session,
	}
	for index := 0; index < maximum; index++ {
		row, err := dflashLogitRow(logits, index+1, target.vocab.Len())
		if err != nil {
			return nil, err
		}
		token, probability, err := greedyLogit(row)
		if err != nil {
			return nil, err
		}
		if probability < minimumProbability {
			break
		}
		id := tokenizer.TokenID(token)
		draft.Tokens = append(draft.Tokens, id)
		draft.Probabilities = append(draft.Probabilities, probability)
		if target.vocab.IsEOG(id) {
			break
		}
	}
	return draft, nil
}

// VerifyDFlashGreedy: target verification and feature-cache resync.
func (r *Runner) VerifyDFlashGreedy(
	ctx context.Context,
	target *Runner,
	draft *DFlashDraft,
) (*DFlashVerification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) || !validDFlashSession(draft.Base) {
		return nil, errors.New("inference: DFlash verification inputs are invalid")
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *DFlashSession,
	) (reference.Value, *DFlashSession, error) {
		return r.advanceDFlashVerification(ctx, target, token, state)
	})
}

func validDFlashSession(session *DFlashSession) bool {
	return session != nil && session.Cache != nil && session.TargetCache != nil &&
		len(session.TargetTokens) > 0 && session.Position == uint32(len(session.TargetTokens)) &&
		session.Cache.Position == session.Position &&
		effectiveCachePosition(session.TargetCache) == session.Position
}

func (r *Runner) advanceDFlashVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	session *DFlashSession,
) (reference.Value, *DFlashSession, error) {
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
	fused, err := r.FuseDFlashFeatures(ctx, features)
	if err != nil {
		return reference.Value{}, nil, err
	}
	cache, err := r.InjectDFlashFeatures(ctx, fused, []uint32{session.Position}, session.Cache)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &DFlashSession{
		Cache: cache, TargetCache: targetCache, TargetTokens: tokens,
		Position: session.Position + 1,
	}
	if !validDFlashSession(next) {
		return reference.Value{}, nil, errors.New("inference: DFlash verification cache resync failed")
	}
	return logits, next, nil
}

func dflashLogitRow(logits reference.Value, row int, vocabulary int) ([]float32, error) {
	if logits.Shape.Rank != 2 || vocabulary <= 0 || logits.Shape.Dims[0] != uint64(vocabulary) ||
		row < 0 || row >= int(logits.Shape.Dims[1]) || len(logits.Data) != vocabulary*int(logits.Shape.Dims[1]) {
		return nil, errors.New("inference: DFlash block logits are incompatible")
	}
	return logits.Data[row*vocabulary : (row+1)*vocabulary], nil
}
