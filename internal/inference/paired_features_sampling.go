package inference

import (
	"context"
	"errors"

	"overgo/internal/sampling"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// PairedFeatureSampledDraft: block proposals plus sampler checkpoints.
type PairedFeatureSampledDraft = sampledDraft[*PairedFeatureSession]

// DraftPairedFeatureSampled: transactional masked-block sampling.
func (r *Runner) DraftPairedFeatureSampled(
	ctx context.Context,
	target *Runner,
	session *PairedFeatureSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *PairedFeatureSampledDraft, err error) {
	if r == nil || target == nil || !validPairedFeatureSession(session) || sampler == nil || len(history) == 0 ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: paired-feature sampled draft inputs are invalid")
	}
	initialToken := history[len(history)-1]
	logits, err := r.DraftPairedFeatureBlock(ctx, target, initialToken, maximum, session.Cache)
	if err != nil {
		return nil, err
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "paired-feature", target.vocab.IsEOG,
		func(_ tokenizer.TokenID, state *PairedFeatureSession, index int) ([]float32, *PairedFeatureSession, error) {
			row, rowErr := pairedFeatureLogitRow(logits, index+1, target.vocab.Len())
			return row, state, rowErr
		},
	)
}

// VerifyPairedFeatureSampled: ratio verification plus sampler rollback.
func (r *Runner) VerifyPairedFeatureSampled(
	ctx context.Context,
	target *Runner,
	draft *PairedFeatureSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *PairedFeatureVerification, err error) {
	if r == nil || target == nil || draft == nil || !validPairedFeatureSession(draft.Base) ||
		!validVerificationSamplers(draftSampler, targetSampler) {
		return nil, errors.New("inference: paired-feature sampled verification inputs are invalid")
	}
	if !validSampledDraft(draft) {
		return nil, errors.New("inference: paired-feature sampled draft state is inconsistent")
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, "paired-feature",
		func(token tokenizer.TokenID, state *PairedFeatureSession) (reference.Value, *PairedFeatureSession, error) {
			return r.advancePairedFeatureVerification(ctx, target, token, state)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *PairedFeatureSession) (*PairedFeatureVerification, error) {
			return &PairedFeatureVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
}
