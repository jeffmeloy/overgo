package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Qwen35MTPDraft: greedy proposals from one target-selected token.
type Qwen35MTPDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Base          *Qwen35MTPSession
}

// Qwen35MTPVerification: accepted prefix plus target correction.
type Qwen35MTPVerification struct {
	Accepted     int
	NextToken    tokenizer.TokenID
	TargetLogits reference.Value
	Session      *Qwen35MTPSession
}

// DraftQwen35MTPGreedy: bounded high-confidence proposal chain.
func (r *Runner) DraftQwen35MTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *Qwen35MTPSession,
	maximum int,
	minimumProbability float64,
) (*Qwen35MTPDraft, error) {
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 ||
		math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Qwen3.5 MTP draft limits are invalid")
	}
	if session == nil {
		return nil, errors.New("inference: Qwen3.5 MTP draft session is nil")
	}
	draft := &Qwen35MTPDraft{
		InitialToken:  initialToken,
		Tokens:        make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum),
		Base:          session,
	}
	currentToken := initialToken
	currentSession := session
	for range maximum {
		logits, next, err := r.AdvanceQwen35MTP(ctx, currentToken, currentSession)
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
		currentToken = tokenizer.TokenID(token)
		currentSession = next
		if r.vocab.IsEOG(currentToken) {
			break
		}
	}
	return draft, nil
}

// VerifyQwen35MTPGreedy: target check, rollback, and hidden-state resync.
func (r *Runner) VerifyQwen35MTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *Qwen35MTPDraft,
) (*Qwen35MTPVerification, error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil {
		return nil, errors.New("inference: Qwen3.5 MTP verification inputs are invalid")
	}
	if len(draft.Tokens) != len(draft.Probabilities) {
		return nil, errors.New("inference: Qwen3.5 MTP draft state is inconsistent")
	}
	if r.weights.Qwen35MTP != nil && r.weights.Qwen35MTP.MTPOnly {
		if err := r.validateQwen35MTPTarget(target); err != nil {
			return nil, err
		}
	} else if r != target {
		return nil, errors.New("inference: bundled Qwen3.5 MTP verification requires its owning target runner")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: Qwen3.5 MTP session belongs to a different target model")
	}
	mtpSession := draft.Base
	targetCache := draft.Base.TrunkCache
	currentToken := draft.InitialToken
	accepted := 0
	for {
		hidden, nextTargetCache, err := target.ForwardCached(
			ctx, []tokenizer.TokenID{currentToken}, targetCache,
		)
		if err != nil {
			return nil, err
		}
		logits, err := target.qwen35MTPProjectLogits(ctx, hidden)
		if err != nil {
			return nil, err
		}
		nextToken, _, err := greedyLogit(logits.Data)
		if err != nil {
			return nil, err
		}
		_, nextMTPSession, err := r.AdvanceQwen35MTP(ctx, currentToken, mtpSession)
		if err != nil {
			return nil, err
		}
		nextMTPSession.PendingHidden = lastHiddenColumn(hidden)
		nextMTPSession.TrunkCache = nextTargetCache
		mtpSession = nextMTPSession
		targetCache = nextTargetCache
		if accepted >= len(draft.Tokens) || tokenizer.TokenID(nextToken) != draft.Tokens[accepted] {
			return &Qwen35MTPVerification{
				Accepted: accepted, NextToken: tokenizer.TokenID(nextToken),
				TargetLogits: logits, Session: mtpSession,
			}, nil
		}
		currentToken = draft.Tokens[accepted]
		accepted++
	}
}

func (r *Runner) qwen35MTPProjectLogits(
	ctx context.Context,
	hidden reference.Value,
) (reference.Value, error) {
	return r.projectHiddenLogits(ctx, hidden)
}

func (r *Runner) projectHiddenLogits(
	ctx context.Context,
	hidden reference.Value,
) (reference.Value, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: Qwen3.5 MTP target runner is unavailable")
	}
	return r.projectAllLogits(ctx, hidden)
}

func greedyLogit(logits []float32) (int, float64, error) {
	if len(logits) == 0 {
		return 0, 0, errors.New("inference: greedy logits are empty")
	}
	best := 0
	maximum := float64(logits[0])
	if math.IsNaN(maximum) {
		return 0, 0, errors.New("inference: greedy logits contain NaN")
	}
	for index := 1; index < len(logits); index++ {
		value := float64(logits[index])
		if math.IsNaN(value) {
			return 0, 0, errors.New("inference: greedy logits contain NaN")
		}
		if value > maximum {
			maximum, best = value, index
		}
	}
	if math.IsInf(maximum, -1) {
		return 0, 0, errors.New("inference: greedy logits contain no finite candidate")
	}
	var denominator float64
	for _, raw := range logits {
		value := float64(raw)
		if math.IsInf(value, 1) {
			if math.IsInf(maximum, 1) && value == maximum {
				denominator++
			}
			continue
		}
		denominator += math.Exp(value - maximum)
	}
	if denominator == 0 || math.IsNaN(denominator) {
		return 0, 0, fmt.Errorf("inference: greedy probability normalization failed")
	}
	probability := 1 / denominator
	return best, probability, nil
}
