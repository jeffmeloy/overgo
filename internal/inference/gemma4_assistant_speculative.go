package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Gemma4AssistantDraft: bounded greedy proposals.
type Gemma4AssistantDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Base          *Gemma4AssistantSession
}

// Gemma4AssistantVerification: accepted prefix plus correction.
type Gemma4AssistantVerification struct {
	Accepted     int
	NextToken    tokenizer.TokenID
	TargetLogits reference.Value
	Session      *Gemma4AssistantSession
}

// DraftGemma4AssistantGreedy: bounded confident proposals.
func (r *Runner) DraftGemma4AssistantGreedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *Gemma4AssistantSession,
	maximum int,
	minimumProbability float64,
) (*Gemma4AssistantDraft, error) {
	if r == nil || target == nil || session == nil || maximum <= 0 ||
		minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Gemma 4 assistant draft inputs are invalid")
	}
	draft := &Gemma4AssistantDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum), Base: session,
	}
	currentToken, currentSession := initialToken, session
	for range maximum {
		logits, next, err := r.AdvanceGemma4Assistant(ctx, target, currentToken, currentSession)
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
		if target.vocab.IsEOG(currentToken) {
			break
		}
	}
	return draft, nil
}

// VerifyGemma4AssistantGreedy: target verification and hidden resync.
func (r *Runner) VerifyGemma4AssistantGreedy(
	ctx context.Context,
	target *Runner,
	draft *Gemma4AssistantDraft,
) (*Gemma4AssistantVerification, error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil || draft.Base.TargetCache == nil {
		return nil, errors.New("inference: Gemma 4 assistant verification inputs are invalid")
	}
	if len(draft.Tokens) != len(draft.Probabilities) {
		return nil, errors.New("inference: Gemma 4 assistant draft state is inconsistent")
	}
	if err := r.validateGemma4AssistantTarget(target); err != nil {
		return nil, err
	}
	assistantSession := draft.Base
	targetCache := draft.Base.TargetCache
	currentToken := draft.InitialToken
	accepted := 0
	for {
		hidden, nextTargetCache, err := target.ForwardCached(ctx, []tokenizer.TokenID{currentToken}, targetCache)
		if err != nil {
			return nil, err
		}
		logits, err := target.projectHiddenLogits(ctx, hidden)
		if err != nil {
			return nil, err
		}
		nextToken, _, err := greedyLogit(logits.Data)
		if err != nil {
			return nil, err
		}
		_, nextAssistantSession, err := r.AdvanceGemma4Assistant(ctx, target, currentToken, assistantSession)
		if err != nil {
			return nil, err
		}
		nextAssistantSession.TargetCache = nextTargetCache
		nextAssistantSession.PendingHidden = lastHiddenColumn(hidden)
		nextAssistantSession.Position = effectiveCachePosition(nextTargetCache)
		assistantSession, targetCache = nextAssistantSession, nextTargetCache
		if accepted >= len(draft.Tokens) || tokenizer.TokenID(nextToken) != draft.Tokens[accepted] {
			return &Gemma4AssistantVerification{
				Accepted: accepted, NextToken: tokenizer.TokenID(nextToken),
				TargetLogits: logits, Session: assistantSession,
			}, nil
		}
		currentToken = draft.Tokens[accepted]
		accepted++
	}
}
