package inference

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Cohere2MTPSession: trunk snapshot plus draft state.
type Cohere2MTPSession = Qwen35MTPSession

// NewCohere2MTPSession: bundled trunk-prefix setup.
func (r *Runner) NewCohere2MTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*Cohere2MTPSession, error) {
	if r == nil || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Cohere2-MoE MTP inputs are invalid")
	}
	if err := r.validateCohere2MTP(); err != nil {
		return nil, err
	}
	if r.weights.Cohere2MTP.MTPOnly {
		return nil, errors.New("inference: Cohere2-MoE MTP-only model requires NewCohere2MTPPairedSession")
	}
	return r.newSingleHeadMTPSession(ctx, tokenIDs)
}

// NewCohere2MTPPairedSession: sidecar/target setup.
func (r *Runner) NewCohere2MTPPairedSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*Cohere2MTPSession, error) {
	if r == nil || target == nil || r == target || r.path == target.path || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Cohere2-MoE MTP sidecar and target inputs are invalid")
	}
	if err := r.validateCohere2MTPTarget(target); err != nil {
		return nil, err
	}
	return target.newSingleHeadMTPSession(ctx, tokenIDs)
}

// AdvanceCohere2MTP: one autoregressive draft step.
func (r *Runner) AdvanceCohere2MTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *Cohere2MTPSession,
) (reference.Value, *Cohere2MTPSession, error) {
	if r == nil || session == nil || session.TrunkCache == nil {
		return reference.Value{}, nil, errors.New("inference: Cohere2-MoE MTP session is invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: Cohere2-MoE MTP runner is unavailable")
	}
	if err := r.validateCohere2MTP(); err != nil {
		return reference.Value{}, nil, err
	}
	if err := r.validateCohere2MTPSession(session); err != nil {
		return reference.Value{}, nil, err
	}
	if tokenID < 0 || int(tokenID) >= r.vocab.Len() {
		return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	mtp := r.weights.Cohere2MTP
	return r.advanceSingleHeadMTP(ctx, tokenID, session, singleHeadMTPAdapter{
		nodePrefix: "cohere2_mtp", layer: mtp.Layer,
		embeddingNorm: mtp.EmbeddingNorm, hiddenNorm: mtp.HiddenNorm, project: mtp.EHProjection,
		tokenEmbedding: mtp.TokenEmbedding, outputNorm: mtp.OutputNorm, output: mtp.Output,
		buildInput: model.BuildCohere2MTPInput,
		buildBlock: func(builder *tensor.Builder, input *tensor.Tensor, spec model.Spec, weights model.LayerGraphWeights, positions []uint32, pastKey, pastValue *tensor.Tensor) (singleHeadMTPBlock, error) {
			block, err := model.BuildCohere2MTPBlockCached(builder, input, spec, weights, positions, pastKey, pastValue)
			return singleHeadMTPBlock{output: block.Output, key: block.Key, value: block.Value}, err
		},
		buildOutputs: model.BuildCohere2MTPOutputs,
	})
}

func (r *Runner) validateCohere2MTP() error {
	if r.weights.Cohere2MTP == nil || !r.hasDraftSession(model.DraftCohere2MTP, 1) {
		return errors.New("inference: model has no supported Cohere2-MoE MTP block")
	}
	return nil
}

func (r *Runner) validateCohere2MTPSession(session *Cohere2MTPSession) error {
	return r.validateSingleHeadMTPSession(session, "Cohere2-MoE MTP", true)
}

func (r *Runner) validateCohere2MTPTarget(target *Runner) error {
	if err := r.validateCohere2MTP(); err != nil {
		return err
	}
	targetMTPOnly := target != nil && target.weights.Cohere2MTP != nil && target.weights.Cohere2MTP.MTPOnly
	if err := r.validateSingleHeadMTPTarget(
		target, r.weights.Cohere2MTP.MTPOnly, targetMTPOnly, "Cohere2-MoE MTP",
	); err != nil {
		return err
	}
	return nil
}
