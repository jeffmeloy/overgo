package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Eagle3SampledDraft: proposals plus sampler checkpoints.
type Eagle3SampledDraft = sampledDraft[*Eagle3Session]

// DraftEagle3Sampled: transactional sampled proposals.
func (r *Runner) DraftEagle3Sampled(
	ctx context.Context,
	target *Runner,
	session *Eagle3Session,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *Eagle3SampledDraft, err error) {
	if r == nil || target == nil || session == nil || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: Eagle3 sampled draft inputs are invalid")
	}
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 || math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Eagle3 sampled draft limits are invalid")
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "Eagle3", target.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Eagle3Session, _ int) ([]float32, *Eagle3Session, error) {
			logits, next, advanceErr := r.AdvanceEagle3(ctx, target, token, state)
			return logits.Data, next, advanceErr
		},
	)
}

// VerifyEagle3Sampled: ratio verification plus rollback.
func (r *Runner) VerifyEagle3Sampled(
	ctx context.Context,
	target *Runner,
	draft *Eagle3SampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *Eagle3Verification, err error) {
	if r == nil || target == nil || draft == nil || !validEagle3CoordinatorSession(draft.Base) ||
		draftSampler == nil || targetSampler == nil || draftSampler == targetSampler {
		return nil, errors.New("inference: Eagle3 sampled verification inputs are invalid")
	}
	if !validSampledDraft(draft) {
		return nil, errors.New("inference: Eagle3 sampled draft state is inconsistent")
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, "Eagle3",
		func(token tokenizer.TokenID, state *Eagle3Session) (reference.Value, *Eagle3Session, error) {
			return r.advanceEagle3Verification(ctx, target, token, state)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *Eagle3Session) (*Eagle3Verification, error) {
			return &Eagle3Verification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
}
