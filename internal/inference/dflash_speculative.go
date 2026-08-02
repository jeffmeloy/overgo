package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// DFlashDraft: one masked-block proposal.
type DFlashDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Base          *DFlashSession
}

// DFlashVerification: accepted prefix plus target correction.
type DFlashVerification struct {
	Accepted     int
	NextToken    tokenizer.TokenID
	TargetLogits reference.Value
	Session      *DFlashSession
}

// DraftDFlashGreedy: bounded masked-block proposals.
func (r *Runner) DraftDFlashGreedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *DFlashSession,
	maximum int,
	minimumProbability float64,
) (*DFlashDraft, error) {
	if r == nil || target == nil || !validDFlashSession(session) || maximum <= 0 ||
		minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: DFlash draft inputs are invalid")
	}
	logits, err := r.DraftDFlashBlock(ctx, target, initialToken, maximum, session.Cache)
	if err != nil {
		return nil, err
	}
	draft := &DFlashDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum), Base: session,
	}
	for index := 0; index < maximum; index++ {
		row, err := dflashLogitRow(logits, index+1, target.vocab.Len())
		if err != nil {
			return nil, err
		}
		token, probability, err := greedyLogit(row)
		if err != nil {
			return nil, err
		}
		if probability < minimumProbability {
			break
		}
		id := tokenizer.TokenID(token)
		draft.Tokens = append(draft.Tokens, id)
		draft.Probabilities = append(draft.Probabilities, probability)
		if target.vocab.IsEOG(id) {
			break
		}
	}
	return draft, nil
}

// VerifyDFlashGreedy: target verification and feature-cache resync.
func (r *Runner) VerifyDFlashGreedy(
	ctx context.Context,
	target *Runner,
	draft *DFlashDraft,
) (*DFlashVerification, error) {
	if r == nil || target == nil || draft == nil || !validDFlashSession(draft.Base) {
		return nil, errors.New("inference: DFlash verification inputs are invalid")
	}
	if len(draft.Tokens) != len(draft.Probabilities) {
		return nil, errors.New("inference: DFlash draft state is inconsistent")
	}
	session := draft.Base
	currentToken := draft.InitialToken
	accepted := 0
	for {
		logits, next, err := r.advanceDFlashVerification(ctx, target, currentToken, session)
		if err != nil {
			return nil, err
		}
		session = next
		nextToken, _, err := greedyLogit(logits.Data)
		if err != nil {
			return nil, err
		}
		if accepted >= len(draft.Tokens) || tokenizer.TokenID(nextToken) != draft.Tokens[accepted] {
			return &DFlashVerification{
				Accepted: accepted, NextToken: tokenizer.TokenID(nextToken),
				TargetLogits: logits, Session: session,
			}, nil
		}
		currentToken = draft.Tokens[accepted]
		accepted++
	}
}

func validDFlashSession(session *DFlashSession) bool {
	return session != nil && session.Cache != nil && session.TargetCache != nil &&
		len(session.TargetTokens) > 0 && session.Position == uint32(len(session.TargetTokens)) &&
		session.Cache.Position == session.Position &&
		effectiveCachePosition(session.TargetCache) == session.Position
}

func (r *Runner) advanceDFlashVerification(
	ctx context.Context,
	target *Runner,
	currentToken tokenizer.TokenID,
	session *DFlashSession,
) (reference.Value, *DFlashSession, error) {
	hidden, targetCache, err := target.ForwardCached(ctx, []tokenizer.TokenID{currentToken}, session.TargetCache)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, err := target.projectHiddenLogits(ctx, hidden)
	if err != nil {
		return reference.Value{}, nil, err
	}
	tokens := append(append([]tokenizer.TokenID(nil), session.TargetTokens...), currentToken)
	cache, err := r.SyncDFlashPrefix(ctx, target, tokens, session.Cache)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &DFlashSession{
		Cache: cache, TargetCache: targetCache, TargetTokens: tokens,
		Position: session.Position + 1,
	}
	if !validDFlashSession(next) {
		return reference.Value{}, nil, errors.New("inference: DFlash verification cache resync failed")
	}
	return logits, next, nil
}

func dflashLogitRow(logits reference.Value, row int, vocabulary int) ([]float32, error) {
	if logits.Shape.Rank != 2 || vocabulary <= 0 || logits.Shape.Dims[0] != uint64(vocabulary) ||
		row < 0 || row >= int(logits.Shape.Dims[1]) || len(logits.Data) != vocabulary*int(logits.Shape.Dims[1]) {
		return nil, errors.New("inference: DFlash block logits are incompatible")
	}
	return logits.Data[row*vocabulary : (row+1)*vocabulary], nil
}
