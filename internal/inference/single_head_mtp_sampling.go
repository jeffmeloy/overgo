package inference

import (
	"context"
	"errors"

	"overgo/internal/sampling"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type MTPSampledDraft = sampledDraft[*MTPSession]

func (r *Runner) DraftMTPSampled(
	ctx context.Context,
	session *MTPSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (*MTPSampledDraft, error) {
	if r == nil || session == nil || sampler == nil || len(history) == 0 ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: MTP sampled draft inputs are invalid")
	}
	plan, _, err := r.singleHeadMTP()
	if err != nil {
		return nil, err
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, plan.Label, r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *MTPSession, _ int) ([]float32, *MTPSession, error) {
			logits, next, advanceErr := r.AdvanceMTP(ctx, token, state)
			return logits.Data, next, advanceErr
		},
	)
}

func (r *Runner) VerifyMTPSampled(
	ctx context.Context,
	target *Runner,
	draft *MTPSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (*MTPVerification, error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil ||
		!validVerificationSamplers(draftSampler, targetSampler) || !validSampledDraft(draft) {
		return nil, errors.New("inference: MTP sampled verification inputs are invalid")
	}
	if err := r.validateMTPVerificationTarget(target, draft.Base); err != nil {
		return nil, err
	}
	plan, _, err := r.singleHeadMTP()
	if err != nil {
		return nil, err
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, plan.Label,
		func(token tokenizer.TokenID, state *MTPSession) (reference.Value, *MTPSession, error) {
			return r.advanceMTPVerification(ctx, target, token, state, state.TrunkCache)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *MTPSession) (*MTPVerification, error) {
			return &MTPVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
}

func (r *Runner) validateMTPVerificationTarget(target *Runner, session *MTPSession) error {
	plan, catalog, err := r.singleHeadMTP()
	if err != nil {
		return err
	}
	return r.validateSingleHeadMTPVerificationTarget(
		target, session, catalog.MTPOnly, plan.Label, r.validateMTPTarget,
	)
}
