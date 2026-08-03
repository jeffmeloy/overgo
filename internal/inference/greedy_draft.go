package inference

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum), Base: session,
	}
	currentToken, currentSession := initialToken, session
	for range maximum {
		logits, next, err := advance(currentToken, currentSession)
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
	accepted := 0
	for {
		logits, next, err := advance(currentToken, state)
		if err != nil {
			return nil, err
		}
		state = next
		nextToken, _, err := greedyLogit(logits.Data)
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
	return best, 1 / denominator, nil
}
