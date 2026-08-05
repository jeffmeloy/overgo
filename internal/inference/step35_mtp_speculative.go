package inference

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Step35MTPDraft: bounded proposals across trained heads.
type Step35MTPDraft = greedyDraft[*Step35MTPSession]

// Step35MTPVerification: accepted prefix plus target correction.
type Step35MTPVerification = greedyVerification[*Step35MTPSession]

type HYV3MTPDraft = Step35MTPDraft
type HYV3MTPVerification = Step35MTPVerification

// DraftStep35MTPGreedy: high-confidence trained-head chain.
func (r *Runner) DraftStep35MTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *Step35MTPSession,
	maximum int,
	minimumProbability float64,
) (*Step35MTPDraft, error) {
	if err := r.validateStep35MTP(); err != nil {
		return nil, err
	}
	return r.draftMultiHeadMTPGreedy(ctx, initialToken, session, maximum, minimumProbability)
}

// DraftHYV3MTPGreedy: HY-V3 trained-head proposals.
func (r *Runner) DraftHYV3MTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *HYV3MTPSession,
	maximum int,
	minimumProbability float64,
) (*HYV3MTPDraft, error) {
	if err := r.validateHYV3MTP(); err != nil {
		return nil, err
	}
	return r.draftMultiHeadMTPGreedy(ctx, initialToken, session, maximum, minimumProbability)
}

func (r *Runner) draftMultiHeadMTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *Step35MTPSession,
	maximum int,
	minimumProbability float64,
) (*Step35MTPDraft, error) {
	if r == nil || session == nil || len(session.DraftTokens) != 0 ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: Step3.5 MTP draft inputs are invalid")
	}
	remaining := len(r.multiHeadMTPWeights())
	if remaining <= 0 {
		return nil, errors.New("inference: Step3.5 MTP head chain is exhausted")
	}
	maximum = min(maximum, remaining)
	return draftGreedy(
		initialToken, session, min(maximum, remaining), minimumProbability, r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Step35MTPSession) (reference.Value, *Step35MTPSession, error) {
			return r.advanceMultiHeadMTP(ctx, token, state)
		},
	)
}

// VerifyStep35MTPGreedy: target check plus head-cache resync.
func (r *Runner) VerifyStep35MTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *Step35MTPDraft,
) (*Step35MTPVerification, error) {
	if err := r.validateStep35MTP(); err != nil {
		return nil, err
	}
	return r.verifyMultiHeadMTPGreedy(ctx, target, draft)
}

// VerifyHYV3MTPGreedy: HY-V3 target check and resync.
func (r *Runner) VerifyHYV3MTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *HYV3MTPDraft,
) (*HYV3MTPVerification, error) {
	if err := r.validateHYV3MTP(); err != nil {
		return nil, err
	}
	return r.verifyMultiHeadMTPGreedy(ctx, target, draft)
}

func (r *Runner) verifyMultiHeadMTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *Step35MTPDraft,
) (*Step35MTPVerification, error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil {
		return nil, errors.New("inference: Step3.5 MTP verification inputs are invalid")
	}
	if r != target {
		return nil, errors.New("inference: bundled Step3.5 MTP verification requires its owning target runner")
	}
	if len(draft.Tokens) != len(draft.Probabilities) || len(draft.Tokens) > len(r.multiHeadMTPWeights()) {
		return nil, errors.New("inference: Step3.5 MTP draft state is inconsistent")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: Step3.5 MTP session belongs to a different target model")
	}
	targetCache := draft.Base.TrunkCache
	currentToken := draft.InitialToken
	processedTokens := make([]tokenizer.TokenID, 0, len(draft.Tokens)+1)
	processedHidden := make([]reference.Value, 0, len(draft.Tokens)+1)
	accepted := 0
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
		nextToken, _, sampleErr := greedyLogit(logits.Data)
		if sampleErr != nil {
			return nil, sampleErr
		}
		if accepted >= len(draft.Tokens) || tokenizer.TokenID(nextToken) != draft.Tokens[accepted] {
			session, syncErr := r.resyncStep35MTPSession(
				ctx, draft.Base, targetCache, processedTokens, processedHidden,
			)
			if syncErr != nil {
				return nil, syncErr
			}
			return &Step35MTPVerification{
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
		return reference.Value{}, reference.Value{}, nil, errors.New("inference: Step3.5 target runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, reference.Value{}, nil, errors.New("inference: Step3.5 target runner is unavailable")
	}
	var hidden reference.Value
	var nextCache *KVCache
	var err error
	if r.usesStep35MTPGraph() {
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
	if r.usesStep35MTPGraph() {
		normalized, err = r.runOutputNorm(ctx, hidden)
		if err != nil {
			return reference.Value{}, reference.Value{}, nil, err
		}
	}
	logits, err := r.projectAllLogits(ctx, normalized)
	if err != nil {
		return reference.Value{}, reference.Value{}, nil, err
	}
	lastHidden, err := lastValueColumn(hidden)
	if err != nil {
		return reference.Value{}, reference.Value{}, nil, err
	}
	return logits, lastHidden, nextCache, nil
}

func (r *Runner) resyncStep35MTPSession(
	ctx context.Context,
	base *Step35MTPSession,
	trunkCache *KVCache,
	tokens []tokenizer.TokenID,
	targetHidden []reference.Value,
) (*Step35MTPSession, error) {
	if r == nil || base == nil || trunkCache == nil || len(tokens) == 0 || len(tokens) != len(targetHidden) {
		return nil, errors.New("inference: Step3.5 MTP resync inputs are invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: Step3.5 MTP runner is unavailable")
	}
	width := int(r.spec.EmbeddingLength)
	hidden := reference.Value{
		Shape: tensor.MustShape(uint64(width), uint64(len(tokens))),
		Data:  make([]float32, width*len(tokens)),
	}
	copy(hidden.Data, base.PendingHidden.Data)
	for index := 1; index < len(tokens); index++ {
		copy(hidden.Data[index*width:], targetHidden[index-1].Data)
	}
	positions := tokenPositions(base.MTPStart, len(tokens))
	heads := make([]LayerCache, len(base.Heads))
	for offset := range heads {
		_, _, headCache, err := r.runMultiHeadMTPHeadLocked(
			ctx, tokens, hidden, positions, &base.Heads[offset], uint32(offset),
		)
		if err != nil {
			return nil, fmt.Errorf("inference: Step3.5 MTP head %d resync: %w", offset, err)
		}
		heads[offset] = headCache
	}
	position := effectiveCachePosition(trunkCache)
	return &Step35MTPSession{
		TrunkCache: trunkCache, Heads: heads,
		PendingHidden: targetHidden[len(targetHidden)-1],
		MTPStart:      position, Position: position, targetModel: base.targetModel,
	}, nil
}
