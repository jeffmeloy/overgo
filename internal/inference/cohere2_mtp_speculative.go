package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Cohere2MTPDraft: greedy single-block proposals.
type Cohere2MTPDraft = Qwen35MTPDraft

// Cohere2MTPVerification: accepted prefix plus correction.
type Cohere2MTPVerification = Qwen35MTPVerification

// DraftCohere2MTPGreedy: bounded confident proposals.
func (r *Runner) DraftCohere2MTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *Cohere2MTPSession,
	maximum int,
	minimumProbability float64,
) (*Cohere2MTPDraft, error) {
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 ||
		math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Cohere2-MoE MTP draft limits are invalid")
	}
	if session == nil {
		return nil, errors.New("inference: Cohere2-MoE MTP draft session is nil")
	}
	draft := &Cohere2MTPDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum), Base: session,
	}
	currentToken := initialToken
	currentSession := session
	for range maximum {
		logits, next, err := r.AdvanceCohere2MTP(ctx, currentToken, currentSession)
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
		draft.Tokens = append(draft.Tokens, tokenizer.TokenID(token))
		draft.Probabilities = append(draft.Probabilities, probability)
		currentToken, currentSession = tokenizer.TokenID(token), next
		if r.vocab.IsEOG(currentToken) {
			break
		}
	}
	return draft, nil
}

// VerifyCohere2MTPGreedy: target check plus hidden resync.
func (r *Runner) VerifyCohere2MTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *Cohere2MTPDraft,
) (*Cohere2MTPVerification, error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil {
		return nil, errors.New("inference: Cohere2-MoE MTP verification inputs are invalid")
	}
	if len(draft.Tokens) != len(draft.Probabilities) {
		return nil, errors.New("inference: Cohere2-MoE MTP draft state is inconsistent")
	}
	if r.weights.Cohere2MTP != nil && r.weights.Cohere2MTP.MTPOnly {
		if err := r.validateCohere2MTPTarget(target); err != nil {
			return nil, err
		}
	} else if r != target {
		return nil, errors.New("inference: bundled Cohere2-MoE MTP verification requires its owning target runner")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: Cohere2-MoE MTP session belongs to a different target model")
	}
	mtpSession := draft.Base
	targetCache := draft.Base.TrunkCache
	currentToken := draft.InitialToken
	accepted := 0
	for {
		logits, nextSession, err := r.advanceCohere2MTPVerification(
			ctx, target, currentToken, mtpSession, targetCache,
		)
		if err != nil {
			return nil, err
		}
		mtpSession, targetCache = nextSession, nextSession.TrunkCache
		nextToken, _, err := greedyLogit(logits.Data)
		if err != nil {
			return nil, err
		}
		if accepted >= len(draft.Tokens) || tokenizer.TokenID(nextToken) != draft.Tokens[accepted] {
			return &Cohere2MTPVerification{
				Accepted: accepted, NextToken: tokenizer.TokenID(nextToken),
				TargetLogits: logits, Session: mtpSession,
			}, nil
		}
		currentToken = draft.Tokens[accepted]
		accepted++
	}
}

func (r *Runner) advanceCohere2MTPVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	mtpSession *Cohere2MTPSession,
	targetCache *KVCache,
) (reference.Value, *Cohere2MTPSession, error) {
	hidden, nextTargetCache, err := target.ForwardCached(
		ctx, []tokenizer.TokenID{currentToken}, targetCache,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, err := target.cohere2MTPProjectLogits(ctx, hidden)
	if err != nil {
		return reference.Value{}, nil, err
	}
	_, nextSession, err := r.AdvanceCohere2MTP(ctx, currentToken, mtpSession)
	if err != nil {
		return reference.Value{}, nil, err
	}
	width := int(hidden.Shape.Dims[0])
	nextSession.PendingHidden = reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), hidden.Data[len(hidden.Data)-width:]...),
	}
	nextSession.TrunkCache = nextTargetCache
	return logits, nextSession, nil
}

func (r *Runner) cohere2MTPProjectLogits(
	ctx context.Context,
	hidden reference.Value,
) (reference.Value, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: Cohere2-MoE target runner is unavailable")
	}
	return r.projectAllLogits(ctx, hidden)
}
