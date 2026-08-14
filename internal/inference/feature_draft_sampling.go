package inference

import (
	"context"
	"errors"

	"overgo/internal/sampling"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// FeatureSampledDraft: proposals plus sampler checkpoints.
type FeatureSampledDraft = sampledDraft[*FeatureDraftSession]

// DraftFeaturesSampled: transactional sampled proposals.
func (r *Runner) DraftFeaturesSampled(
	ctx context.Context,
	target *Runner,
	session *FeatureDraftSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *FeatureSampledDraft, err error) {
	if r == nil || target == nil || !validFeatureDraftSession(session) || sampler == nil || len(history) == 0 ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: feature-draft sampled inputs are invalid")
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "feature-draft", target.vocab.IsEOG,
		func(token tokenizer.TokenID, state *FeatureDraftSession, _ int) ([]float32, *FeatureDraftSession, error) {
			logits, next, advanceErr := r.AdvanceFeatureDraft(ctx, target, token, state)
			return logits.Data, next, advanceErr
		},
	)
}

// VerifyFeatureDraftSampled: ratio verification plus rollback.
func (r *Runner) VerifyFeatureDraftSampled(
	ctx context.Context,
	target *Runner,
	draft *FeatureSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *FeatureDraftVerification, err error) {
	if r == nil || target == nil || draft == nil || !validFeatureDraftSession(draft.Base) ||
		!validVerificationSamplers(draftSampler, targetSampler) {
		return nil, errors.New("inference: feature-draft sampled verification inputs are invalid")
	}
	if !validSampledDraft(draft) {
		return nil, errors.New("inference: feature-draft sampled state is inconsistent")
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, "feature-draft",
		func(token tokenizer.TokenID, state *FeatureDraftSession) (reference.Value, *FeatureDraftSession, error) {
			return r.advanceFeatureDraftVerification(ctx, target, token, state)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *FeatureDraftSession) (*FeatureDraftVerification, error) {
			return &FeatureDraftVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
}
