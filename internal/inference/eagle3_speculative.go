package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Eagle3Draft: bounded greedy proposals.
type Eagle3Draft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Base          *Eagle3Session
}

// Eagle3Verification: accepted prefix plus correction.
type Eagle3Verification struct {
	Accepted     int
	NextToken    tokenizer.TokenID
	TargetLogits reference.Value
	Session      *Eagle3Session
}

// DraftEagle3Greedy: bounded confident proposals.
func (r *Runner) DraftEagle3Greedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *Eagle3Session,
	maximum int,
	minimumProbability float64,
) (*Eagle3Draft, error) {
	if r == nil || target == nil || session == nil || maximum <= 0 ||
		minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Eagle3 draft inputs are invalid")
	}
	draft := &Eagle3Draft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum), Base: session,
	}
	currentToken, currentSession := initialToken, session
	for range maximum {
		logits, next, err := r.AdvanceEagle3(ctx, target, currentToken, currentSession)
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

// VerifyEagle3Greedy: target verification and feature resync.
func (r *Runner) VerifyEagle3Greedy(
	ctx context.Context,
	target *Runner,
	draft *Eagle3Draft,
) (*Eagle3Verification, error) {
	if r == nil || target == nil || draft == nil || !validEagle3CoordinatorSession(draft.Base) {
		return nil, errors.New("inference: Eagle3 verification inputs are invalid")
	}
	if len(draft.Tokens) != len(draft.Probabilities) {
		return nil, errors.New("inference: Eagle3 draft state is inconsistent")
	}
	session := draft.Base
	currentToken := draft.InitialToken
	accepted := 0
	for {
		logits, next, err := r.advanceEagle3Verification(ctx, target, currentToken, session)
		if err != nil {
			return nil, err
		}
		session = next
		nextToken, _, err := greedyLogit(logits.Data)
		if err != nil {
			return nil, err
		}
		if accepted >= len(draft.Tokens) || tokenizer.TokenID(nextToken) != draft.Tokens[accepted] {
			return &Eagle3Verification{
				Accepted: accepted, NextToken: tokenizer.TokenID(nextToken),
				TargetLogits: logits, Session: session,
			}, nil
		}
		currentToken = draft.Tokens[accepted]
		accepted++
	}
}

func validEagle3CoordinatorSession(session *Eagle3Session) bool {
	return session != nil && session.TargetCache != nil && len(session.TargetTokens) > 0 &&
		effectiveCachePosition(session.TargetCache) == uint32(len(session.TargetTokens)) &&
		session.Position+1 == uint32(len(session.TargetTokens))
}

func (r *Runner) advanceEagle3Verification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	session *Eagle3Session,
) (reference.Value, *Eagle3Session, error) {
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
	_, next, err := r.AdvanceEagle3(ctx, target, currentToken, session)
	if err != nil {
		return reference.Value{}, nil, err
	}
	tokens := append(append([]tokenizer.TokenID(nil), session.TargetTokens...), currentToken)
	fused, err := r.FuseEagle3Features(ctx, features)
	if err != nil {
		return reference.Value{}, nil, err
	}
	width := int(fused.Shape.Dims[0])
	next.TargetCache = targetCache
	next.TargetTokens = tokens
	next.PendingFeature = reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), fused.Data[len(fused.Data)-width:]...),
	}
	return logits, next, nil
}
