package inference

import (
	"overgo/internal/sampling"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type greedyDraft[S any] struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Base          S
}

type greedyVerification[S any] struct {
	Accepted     int
	NextToken    tokenizer.TokenID
	TargetLogits reference.Value
	Session      S
}

func draftGreedy[S any](
	initialToken tokenizer.TokenID,
	session S,
	maximum int,
	minimumProbability float64,
	isEOG func(tokenizer.TokenID) bool,
	advance func(tokenizer.TokenID, S) (reference.Value, S, error),
) (*greedyDraft[S], error) {
	draft := &greedyDraft[S]{
		InitialToken: initialToken, Base: session,
	}
	currentToken, currentSession := initialToken, session
	for range maximum {
		logits, next, err := advance(currentToken, currentSession)
		if err != nil {
			return nil, err
		}
		token, probability, err := sampling.GreedyLogit(logits.Data)
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
		if isEOG(currentToken) {
			break
		}
	}
	return draft, nil
}

func verifyGreedy[S any](
	draft *greedyDraft[S],
	advance func(tokenizer.TokenID, S) (reference.Value, S, error),
) (*greedyVerification[S], error) {
	state := draft.Base
	currentToken := draft.InitialToken
	var accepted int
	for {
		logits, next, err := advance(currentToken, state)
		if err != nil {
			return nil, err
		}
		state = next
		nextToken, _, err := sampling.GreedyLogit(logits.Data)
		if err != nil {
			return nil, err
		}
		selected := tokenizer.TokenID(nextToken)
		if accepted >= len(draft.Tokens) || selected != draft.Tokens[accepted] {
			return &greedyVerification[S]{
				Accepted: accepted, NextToken: selected, TargetLogits: logits, Session: state,
			}, nil
		}
		currentToken = draft.Tokens[accepted]
		accepted++
	}
}

func validGreedyDraft[S any](draft *greedyDraft[S]) bool {
	return draft != nil && len(draft.Tokens) == len(draft.Probabilities)
}
