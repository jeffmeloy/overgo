package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/tokenizer"
)

// NextNMTPDraft: greedy dense-tail proposals.
type NextNMTPDraft = Qwen35MTPDraft

// NextNMTPVerification: accepted prefix plus correction.
type NextNMTPVerification = Qwen35MTPVerification

// DraftNextNMTPGreedy: bounded confident proposals.
func (r *Runner) DraftNextNMTPGreedy(
	ctx context.Context,
	initialToken tokenizer.TokenID,
	session *NextNMTPSession,
	maximum int,
	minimumProbability float64,
) (*NextNMTPDraft, error) {
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: NextN MTP draft limits are invalid")
	}
	if session == nil {
		return nil, errors.New("inference: NextN MTP draft session is nil")
	}
	draft := &NextNMTPDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum), Base: session,
	}
	currentToken := initialToken
	currentSession := session
	for range maximum {
		logits, next, err := r.AdvanceNextNMTP(ctx, currentToken, currentSession)
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

// VerifyNextNMTPGreedy: target check and resync.
func (r *Runner) VerifyNextNMTPGreedy(
	ctx context.Context,
	target *Runner,
	draft *NextNMTPDraft,
) (*NextNMTPVerification, error) {
	if r == nil || target == nil || r != target || draft == nil || draft.Base == nil {
		return nil, errors.New("inference: bundled NextN MTP verification inputs are invalid")
	}
	if len(draft.Tokens) != len(draft.Probabilities) {
		return nil, errors.New("inference: NextN MTP draft state is inconsistent")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: NextN MTP session belongs to a different target model")
	}
	mtpSession := draft.Base
	targetCache := draft.Base.TrunkCache
	currentToken := draft.InitialToken
	accepted := 0
	for {
		hidden, nextTargetCache, err := target.ForwardCached(ctx, []tokenizer.TokenID{currentToken}, targetCache)
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
		_, nextMTPSession, err := r.AdvanceNextNMTP(ctx, currentToken, mtpSession)
		if err != nil {
			return nil, err
		}
		nextMTPSession.PendingHidden = lastHiddenColumn(hidden)
		nextMTPSession.TrunkCache = nextTargetCache
		mtpSession, targetCache = nextMTPSession, nextTargetCache
		if accepted >= len(draft.Tokens) || tokenizer.TokenID(nextToken) != draft.Tokens[accepted] {
			return &NextNMTPVerification{
				Accepted: accepted, NextToken: tokenizer.TokenID(nextToken),
				TargetLogits: logits, Session: mtpSession,
			}, nil
		}
		currentToken = draft.Tokens[accepted]
		accepted++
	}
}
