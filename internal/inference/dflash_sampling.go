package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

// DFlashSampledDraft: block proposals plus sampler checkpoints.
type DFlashSampledDraft struct {
	InitialToken  tokenizer.TokenID
	Tokens        []tokenizer.TokenID
	Probabilities []float64
	Distributions [][]sampling.TokenProbability
	History       []tokenizer.TokenID
	Base          *DFlashSession
	samplerStates [][]byte
}

// DraftDFlashSampled: transactional masked-block sampling.
func (r *Runner) DraftDFlashSampled(
	ctx context.Context,
	target *Runner,
	session *DFlashSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *DFlashSampledDraft, err error) {
	if r == nil || target == nil || !validDFlashSession(session) || sampler == nil || len(history) == 0 ||
		maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: DFlash sampled draft inputs are invalid")
	}
	initialState, err := sampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save DFlash draft sampler: %w", err)
	}
	defer func() {
		restoreErr := sampler.LoadState(initialState)
		if err == nil && restoreErr != nil {
			draft = nil
			err = fmt.Errorf("inference: restore DFlash draft sampler: %w", restoreErr)
		}
	}()
	initialToken := history[len(history)-1]
	logits, err := r.DraftDFlashBlock(ctx, target, initialToken, maximum, session.Cache)
	if err != nil {
		return nil, err
	}
	draft = &DFlashSampledDraft{
		InitialToken: initialToken, Tokens: make([]tokenizer.TokenID, 0, maximum),
		Probabilities: make([]float64, 0, maximum),
		Distributions: make([][]sampling.TokenProbability, 0, maximum),
		History:       append([]tokenizer.TokenID(nil), history...), Base: session,
		samplerStates: [][]byte{append([]byte(nil), initialState...)},
	}
	currentHistory := tokenIDsAsInts(history)
	for index := 0; index < maximum; index++ {
		row, rowErr := dflashLogitRow(logits, index+1, target.vocab.Len())
		if rowErr != nil {
			return nil, rowErr
		}
		result, sampleErr := sampler.SampleWithHistoryProbabilities(row, currentHistory, len(row))
		if sampleErr != nil {
			return nil, sampleErr
		}
		if result.SelectedProbability < minimumProbability {
			break
		}
		state, stateErr := sampler.SaveState()
		if stateErr != nil {
			return nil, fmt.Errorf("inference: save DFlash draft checkpoint: %w", stateErr)
		}
		token := tokenizer.TokenID(result.Token)
		draft.Tokens = append(draft.Tokens, token)
		draft.Probabilities = append(draft.Probabilities, result.SelectedProbability)
		draft.Distributions = append(draft.Distributions, append([]sampling.TokenProbability(nil), result.Top...))
		draft.samplerStates = append(draft.samplerStates, state)
		currentHistory = append(currentHistory, result.Token)
		if target.vocab.IsEOG(token) {
			break
		}
	}
	return draft, nil
}

// VerifyDFlashSampled: ratio verification plus sampler rollback.
func (r *Runner) VerifyDFlashSampled(
	ctx context.Context,
	target *Runner,
	draft *DFlashSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *DFlashVerification, err error) {
	if r == nil || target == nil || draft == nil || !validDFlashSession(draft.Base) ||
		draftSampler == nil || targetSampler == nil || draftSampler == targetSampler {
		return nil, errors.New("inference: DFlash sampled verification inputs are invalid")
	}
	if len(draft.History) == 0 || draft.History[len(draft.History)-1] != draft.InitialToken ||
		len(draft.Tokens) != len(draft.Probabilities) || len(draft.Tokens) != len(draft.Distributions) ||
		len(draft.samplerStates) != len(draft.Tokens)+1 {
		return nil, errors.New("inference: DFlash sampled draft state is inconsistent")
	}
	draftOriginal, err := draftSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save DFlash draft sampler: %w", err)
	}
	targetOriginal, err := targetSampler.SaveState()
	if err != nil {
		return nil, fmt.Errorf("inference: save DFlash target sampler: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = draftSampler.LoadState(draftOriginal)
			_ = targetSampler.LoadState(targetOriginal)
		}
	}()
	if err := draftSampler.LoadState(draft.samplerStates[0]); err != nil {
		return nil, fmt.Errorf("inference: restore DFlash draft base: %w", err)
	}
	session := draft.Base
	currentToken := draft.InitialToken
	history := tokenIDsAsInts(draft.History)
	accepted := 0
	for {
		logits, next, advanceErr := r.advanceDFlashVerification(ctx, target, currentToken, session)
		if advanceErr != nil {
			return nil, advanceErr
		}
		session = next
		var nextToken tokenizer.TokenID
		if accepted < len(draft.Tokens) {
			result, sampleErr := targetSampler.SpeculativeSample(
				logits.Data, history, draft.Distributions[accepted], int(draft.Tokens[accepted]),
			)
			if sampleErr != nil {
				return nil, sampleErr
			}
			nextToken = tokenizer.TokenID(result.Token)
			if result.Accepted {
				accepted++
				currentToken = nextToken
				history = append(history, result.Token)
				continue
			}
		} else {
			token, sampleErr := targetSampler.SampleWithHistory(logits.Data, history)
			if sampleErr != nil {
				return nil, sampleErr
			}
			nextToken = tokenizer.TokenID(token)
		}
		if err := draftSampler.LoadState(draft.samplerStates[accepted]); err != nil {
			return nil, fmt.Errorf("inference: restore DFlash accepted draft: %w", err)
		}
		if err := draftSampler.AcceptToken(int(nextToken)); err != nil {
			return nil, fmt.Errorf("inference: commit DFlash target token to draft sampler: %w", err)
		}
		verification = &DFlashVerification{
			Accepted: accepted, NextToken: nextToken, TargetLogits: logits, Session: session,
		}
		committed = true
		return verification, nil
	}
}
