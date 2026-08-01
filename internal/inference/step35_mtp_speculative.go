package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Step35MTPDraft: bounded proposals across trained heads.
type Step35MTPDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Base          *Step35MTPSession
}

// Step35MTPVerification: accepted prefix plus target correction.
type Step35MTPVerification struct {
	Accepted     int
	NextToken    tokenizer.TokenID
	TargetLogits reference.Value
	Session      *Step35MTPSession
}

// DraftStep35MTPGreedy: high-confidence trained-head chain.
func (r *Runner) DraftStep35MTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *Step35MTPSession,
	maximum int,
	minimumProbability float64,
) (*Step35MTPDraft, error) {
	if r == nil || session == nil || len(session.DraftTokens) != 0 || maximum <= 0 || minimumProbability < 0 ||
		minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Step3.5 MTP draft inputs are invalid")
	}
	remaining := len(r.weights.Step35MTP)
	if remaining <= 0 {
		return nil, errors.New("inference: Step3.5 MTP head chain is exhausted")
	}
	maximum = min(maximum, remaining)
	draft := &Step35MTPDraft{
		InitialToken:  initialToken,
		Tokens:        make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum),
		Base:          session,
	}
	currentToken := initialToken
	currentSession := session
	for range maximum {
		logits, next, err := r.AdvanceStep35MTP(ctx, currentToken, currentSession)
		if err != nil {
			return nil, err
		}
		token, probability, err := greedyLogit(logits.Data)
		if err != nil {
			return nil, err
		}
		if probability < minimumProbability {
			break
		}
		currentToken = tokenizer.TokenID(token)
		draft.Tokens = append(draft.Tokens, currentToken)
		draft.Probabilities = append(draft.Probabilities, probability)
		currentSession = next
		if r.vocab.IsEOG(currentToken) {
			break
		}
	}
	return draft, nil
}

// VerifyStep35MTPGreedy: target check plus head-cache resync.
func (r *Runner) VerifyStep35MTPGreedy(
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
	if len(draft.Tokens) != len(draft.Probabilities) || len(draft.Tokens) > len(r.weights.Step35MTP) {
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
		logits, hidden, nextTargetCache, advanceErr := target.step35TargetAdvance(
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

func (r *Runner) step35TargetAdvance(
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
	hidden, nextCache, err := r.forwardCachedPreOutputNormLocked(
		ctx, []tokenizer.TokenID{tokenID}, cache,
	)
	if err != nil {
		return reference.Value{}, reference.Value{}, nil, err
	}
	normalized, err := r.runOutputNorm(ctx, hidden)
	if err != nil {
		return reference.Value{}, reference.Value{}, nil, err
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
	positions := make([]uint32, len(tokens))
	for index := range positions {
		positions[index] = base.MTPStart + uint32(index)
	}
	heads := make([]LayerCache, len(base.Heads))
	for offset := range heads {
		_, _, headCache, err := r.runStep35MTPHeadLocked(
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
