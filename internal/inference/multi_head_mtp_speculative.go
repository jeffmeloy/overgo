package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// MultiHeadMTPDraft: bounded proposals across trained heads.
type MultiHeadMTPDraft = greedyDraft[*MultiHeadMTPSession]

// MultiHeadMTPVerification: accepted prefix plus target correction.
type MultiHeadMTPVerification = greedyVerification[*MultiHeadMTPSession]

// DraftMultiHeadMTPGreedy runs a high-confidence trained-head chain.
func (r *Runner) DraftMultiHeadMTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *MultiHeadMTPSession,
	maximum int,
	minimumProbability float64,
) (*MultiHeadMTPDraft, error) {
	if _, err := r.multiHeadMTP(); err != nil {
		return nil, err
	}
	if r == nil || session == nil || checked.Nonzero(len(session.DraftTokens)) ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: multi-head MTP draft inputs are invalid")
	}
	remaining := len(session.Heads)
	if !checked.Nonzero(remaining) {
		return nil, errors.New("inference: multi-head MTP head chain is exhausted")
	}
	maximum = min(maximum, remaining)
	return draftGreedy(
		initialToken, session, min(maximum, remaining), minimumProbability, r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *MultiHeadMTPSession) (reference.Value, *MultiHeadMTPSession, error) {
			return r.AdvanceMultiHeadMTP(ctx, token, state)
		},
	)
}

// VerifyMultiHeadMTPGreedy checks the target and resynchronizes head caches.
func (r *Runner) VerifyMultiHeadMTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *MultiHeadMTPDraft,
) (*MultiHeadMTPVerification, error) {
	plan, err := r.multiHeadMTP()
	if err != nil {
		return nil, err
	}
	if r == nil || target == nil || draft == nil || draft.Base == nil {
		return nil, errors.New("inference: multi-head MTP verification inputs are invalid")
	}
	if r != target {
		return nil, errors.New("inference: bundled multi-head MTP verification requires its owning target runner")
	}
	if len(draft.Tokens) != len(draft.Probabilities) || len(draft.Tokens) > int(plan.Heads) {
		return nil, errors.New("inference: multi-head MTP draft state is inconsistent")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: multi-head MTP session belongs to a different target model")
	}
	targetCache := draft.Base.TrunkCache
	currentToken := draft.InitialToken
	processedCapacity := len(draft.Tokens) + tensor.SingletonExtent
	processedTokens := make([]tokenizer.TokenID, 0, processedCapacity)
	processedHidden := make([]reference.Value, 0, processedCapacity)
	var accepted int
	for {
		logits, hidden, nextTargetCache, advanceErr := target.multiHeadMTPTargetAdvance(
			ctx, currentToken, targetCache,
		)
		if advanceErr != nil {
			return nil, advanceErr
		}
		processedTokens = append(processedTokens, currentToken)
		processedHidden = append(processedHidden, hidden)
		targetCache = nextTargetCache
		nextToken, _, sampleErr := sampling.GreedyLogit(logits.Data)
		if sampleErr != nil {
			return nil, sampleErr
		}
		if accepted >= len(draft.Tokens) || tokenizer.TokenID(nextToken) != draft.Tokens[accepted] {
			session, syncErr := r.resyncMultiHeadMTPSession(
				ctx, draft.Base, targetCache, processedTokens, processedHidden,
			)
			if syncErr != nil {
				return nil, syncErr
			}
			return &MultiHeadMTPVerification{
				Accepted: accepted, NextToken: tokenizer.TokenID(nextToken),
				TargetLogits: logits, Session: session,
			}, nil
		}
		currentToken = draft.Tokens[accepted]
		accepted++
	}
}

func (r *Runner) multiHeadMTPTargetAdvance(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	cache *KVCache,
) (reference.Value, reference.Value, *KVCache, error) {
	if r == nil {
		return reference.Value{}, reference.Value{}, nil, errors.New("inference: multi-head target runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, reference.Value{}, nil, errors.New("inference: multi-head target runner is unavailable")
	}
	plan, planErr := r.multiHeadMTP()
	if planErr != nil {
		return reference.Value{}, reference.Value{}, nil, planErr
	}
	var hidden reference.Value
	var nextCache *KVCache
	var err error
	if plan.CarryRawHidden {
		hidden, nextCache, err = r.forwardCachedPreOutputNormLocked(
			ctx, []tokenizer.TokenID{tokenID}, cache,
		)
	} else {
		hidden, nextCache, err = r.forwardCachedLocked(
			ctx, []tokenizer.TokenID{tokenID}, cache,
		)
	}
	if err != nil {
		return reference.Value{}, reference.Value{}, nil, err
	}
	normalized := hidden
	if plan.CarryRawHidden {
		normalized, err = r.runOutputNorm(ctx, hidden)
		if err != nil {
			return reference.Value{}, reference.Value{}, nil, err
		}
	}
	logits, err := r.projectAllLogits(ctx, normalized)
	if err != nil {
		return reference.Value{}, reference.Value{}, nil, err
	}
	lastHidden := hidden.LastRowView().Clone()
	return logits, lastHidden, nextCache, nil
}

func (r *Runner) resyncMultiHeadMTPSession(
	ctx context.Context,
	base *MultiHeadMTPSession,
	trunkCache *KVCache,
	tokens []tokenizer.TokenID,
	targetHidden []reference.Value,
) (*MultiHeadMTPSession, error) {
	if r == nil || base == nil || trunkCache == nil || !checked.Nonzero(len(tokens)) || len(tokens) != len(targetHidden) {
		return nil, errors.New("inference: multi-head MTP resync inputs are invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: multi-head MTP runner is unavailable")
	}
	width := int(r.spec.EmbeddingLength)
	hidden := reference.Value{
		Shape: tensor.MustShape(uint64(width), uint64(len(tokens))),
		Data:  make([]float32, width*len(tokens)),
	}
	copy(hidden.Data, base.PendingHidden.Data)
	precedingHidden, _ := checked.Init(targetHidden)
	for index, row := range precedingHidden {
		copy(hidden.Data[(index+tensor.SingletonExtent)*width:], row.Data)
	}
	positions := tokenPositions(base.MTPStart, len(tokens))
	heads := make([]LayerCache, len(base.Heads))
	for offset := range heads {
		_, _, headCache, err := r.runMultiHeadMTPHeadLocked(
			ctx, tokens, hidden, positions, &base.Heads[offset], uint32(offset),
		)
		if err != nil {
			return nil, fmt.Errorf("inference: multi-head MTP head %d resync: %w", offset, err)
		}
		heads[offset] = headCache
	}
	position := effectiveCachePosition(trunkCache)
	lastSlice, _ := checked.LastSlice(targetHidden)
	lastHidden, _ := checked.First(lastSlice)
	return &MultiHeadMTPSession{
		TrunkCache: trunkCache, Heads: heads,
		PendingHidden: lastHidden,
		MTPStart:      position, Position: position, targetModel: base.targetModel,
	}, nil
}
