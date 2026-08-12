package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// Qwen35MTPSession: trunk snapshot plus independent draft state.
type Qwen35MTPSession struct {
	TrunkCache    *KVCache
	Layer         LayerCache
	PendingHidden reference.Value
	MTPStart      uint32
	Position      uint32
	targetModel   [32]byte
}

// NewQwen35MTPSession: trunk-prefix MTP setup.
func (r *Runner) NewQwen35MTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*Qwen35MTPSession, error) {
	if r == nil || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Qwen3.5 MTP inputs are invalid")
	}
	if err := r.validateQwen35MTP(); err != nil {
		return nil, err
	}
	if r.weights.Qwen35MTP.MTPOnly {
		return nil, errors.New("inference: Qwen3.5 MTP-only model requires NewQwen35MTPPairedSession")
	}
	return r.newSingleHeadMTPSession(ctx, tokenIDs)
}

// NewQwen35MTPPairedSession: sidecar/target prefix setup.
func (r *Runner) NewQwen35MTPPairedSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*Qwen35MTPSession, error) {
	if r == nil || target == nil || r == target || r.path == target.path || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Qwen3.5 MTP sidecar and target inputs are invalid")
	}
	if err := r.validateQwen35MTPTarget(target); err != nil {
		return nil, err
	}
	return target.newSingleHeadMTPSession(ctx, tokenIDs)
}

// AdvanceQwen35MTP: one autoregressive draft step.
func (r *Runner) AdvanceQwen35MTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *Qwen35MTPSession,
) (reference.Value, *Qwen35MTPSession, error) {
	if r == nil || session == nil || session.TrunkCache == nil {
		return reference.Value{}, nil, errors.New("inference: Qwen3.5 MTP session is invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: Qwen3.5 MTP runner is unavailable")
	}
	if err := r.validateQwen35MTP(); err != nil {
		return reference.Value{}, nil, err
	}
	if err := r.validateQwen35MTPSession(session); err != nil {
		return reference.Value{}, nil, err
	}
	if tokenID < 0 || int(tokenID) >= r.vocab.Len() {
		return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	mtp := r.weights.Qwen35MTP
	draftProgram, err := r.draftLayerProgram(0)
	if err != nil {
		return reference.Value{}, nil, err
	}
	return r.advanceSingleHeadMTP(ctx, tokenID, session, singleHeadMTPAdapter{
		nodePrefix: "qwen35_mtp", layer: mtp.Layer, program: draftProgram,
		embeddingNorm: mtp.EmbeddingNorm, hiddenNorm: mtp.HiddenNorm, project: mtp.EHProjection,
		tokenEmbedding: mtp.TokenEmbedding, outputNorm: mtp.OutputNorm, output: mtp.Output,
		buildInput:   model.BuildQwen35MTPInput,
		buildOutputs: model.BuildQwen35MTPOutputs,
	})
}

func (r *Runner) validateQwen35MTP() error {
	if r.weights.Qwen35MTP == nil || !r.hasDraftSession(model.DraftQwen35MTP, 1) {
		return errors.New("inference: model has no supported Qwen3.5 MTP block")
	}
	return nil
}

func (r *Runner) validateQwen35MTPSession(session *Qwen35MTPSession) error {
	return r.validateSingleHeadMTPSession(session, "Qwen3.5 MTP", true)
}

func (r *Runner) validateQwen35MTPTarget(target *Runner) error {
	if err := r.validateQwen35MTP(); err != nil {
		return err
	}
	targetMTPOnly := target != nil && target.weights.Qwen35MTP != nil && target.weights.Qwen35MTP.MTPOnly
	if err := r.validateSingleHeadMTPTarget(
		target, r.weights.Qwen35MTP.MTPOnly, targetMTPOnly, "Qwen3.5 MTP",
	); err != nil {
		return err
	}
	return nil
}
