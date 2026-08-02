package inference

import (
	"context"
	"errors"
	"math"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Cohere2MTPSampledDraft: proposals plus sampler checkpoints.
type Cohere2MTPSampledDraft = Qwen35MTPSampledDraft

// DraftCohere2MTPSampled: transactional sampled proposals.
func (r *Runner) DraftCohere2MTPSampled(
	ctx context.Context,
	session *Cohere2MTPSession,
	sampler *sampling.Sampler,
	history []tokenizer.TokenID,
	maximum int,
	minimumProbability float64,
) (draft *Cohere2MTPSampledDraft, err error) {
	if r == nil || session == nil || sampler == nil || len(history) == 0 {
		return nil, errors.New("inference: Cohere2-MoE MTP sampled draft inputs are invalid")
	}
	if maximum <= 0 || minimumProbability < 0 || minimumProbability > 1 ||
		math.IsNaN(minimumProbability) {
		return nil, errors.New("inference: Cohere2-MoE MTP sampled draft limits are invalid")
	}
	return draftSampled(
		session, sampler, history, maximum, minimumProbability, "Cohere2-MoE MTP", r.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Cohere2MTPSession, _ int) ([]float32, *Cohere2MTPSession, error) {
			logits, next, advanceErr := r.AdvanceCohere2MTP(ctx, token, state)
			return logits.Data, next, advanceErr
		},
	)
}

// VerifyCohere2MTPSampled: ratio verification plus rollback.
func (r *Runner) VerifyCohere2MTPSampled(
	ctx context.Context,
	target *Runner,
	draft *Cohere2MTPSampledDraft,
	draftSampler *sampling.Sampler,
	targetSampler *sampling.Sampler,
) (verification *Cohere2MTPVerification, err error) {
	if r == nil || target == nil || draft == nil || draft.Base == nil ||
		draftSampler == nil || targetSampler == nil || draftSampler == targetSampler {
		return nil, errors.New("inference: Cohere2-MoE MTP sampled verification inputs are invalid")
	}
	if !validSampledDraft(draft) {
		return nil, errors.New("inference: Cohere2-MoE MTP sampled draft state is inconsistent")
	}
	if r.weights.Cohere2MTP != nil && r.weights.Cohere2MTP.MTPOnly {
		if err := r.validateCohere2MTPTarget(target); err != nil {
			return nil, err
		}
	} else if r != target {
		return nil, errors.New("inference: bundled Cohere2-MoE MTP verification requires its owning target runner")
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if draft.Base.targetModel != targetModel {
		return nil, errors.New("inference: Cohere2-MoE MTP session belongs to a different target model")
	}
	return verifySampled(
		draft, draft.Base, draftSampler, targetSampler, "Cohere2-MoE MTP",
		func(token tokenizer.TokenID, state *Cohere2MTPSession) (reference.Value, *Cohere2MTPSession, error) {
			return r.advanceCohere2MTPVerification(ctx, target, token, state, state.TrunkCache)
		},
		func(accepted int, token tokenizer.TokenID, logits reference.Value, state *Cohere2MTPSession) (*Cohere2MTPVerification, error) {
			return &Cohere2MTPVerification{Accepted: accepted, NextToken: token, TargetLogits: logits, Session: state}, nil
		},
	)
}
