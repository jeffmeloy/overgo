package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// DFlashSampledDraft: block proposals plus sampler checkpoints.
type DFlashSampledDraft = sampledDraft[*DFlashSession]

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
	initialToken := history[len(history)-1]
	logits, err := r.DraftDFlashBlock(ctx, target, initialToken, maximum, session.Cache)
	if err != nil {
		return nil, err
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "DFlash", target.vocab.IsEOG,
		func(_ tokenizer.TokenID, state *DFlashSession, index int) ([]float32, *DFlashSession, error) {
			row, rowErr := dflashLogitRow(logits, index+1, target.vocab.Len())
			return row, state, rowErr
		},
	)
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
	if !validSampledDraft(draft) {
		return nil, errors.New("inference: DFlash sampled draft state is inconsistent")
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, "DFlash",
		func(token tokenizer.TokenID, state *DFlashSession) (reference.Value, *DFlashSession, error) {
			return r.advanceDFlashVerification(ctx, target, token, state)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *DFlashSession) (*DFlashVerification, error) {
			return &DFlashVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
}
