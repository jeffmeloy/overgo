package inference

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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
	embeddingInfo := r.weights.TokenEmbedding
	if mtp.TokenEmbedding != nil {
		embeddingInfo = *mtp.TokenEmbedding
	}
	tokenEmbedding, err := r.loadRows(ctx, embeddingInfo, []uint32{uint32(tokenID)})
	if err != nil {
		return reference.Value{}, nil, err
	}
	builder := r.newGraphBuilder()
	tokenInput := builder.Input("qwen35_mtp.token", dtype.F32, tokenEmbedding.Shape)
	hiddenInput := builder.Input("qwen35_mtp.hidden", dtype.F32, session.PendingHidden.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{
		tokenInput: tokenEmbedding, hiddenInput: session.PendingHidden,
	}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	graphWeights, layerFeeds, err := r.qwen35MTPLayerInputs(ctx, builder, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	for node, pointer := range layerFeeds {
		deviceFeeds[node] = pointer
	}
	embeddingNorm, pointer, err := r.deviceOrHostTensor(ctx, builder, mtp.EmbeddingNorm, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[embeddingNorm] = pointer
	}
	hiddenNorm, pointer, err := r.deviceOrHostTensor(ctx, builder, mtp.HiddenNorm, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[hiddenNorm] = pointer
	}
	projection, pointer, err := r.deviceOrHostTensor(ctx, builder, mtp.EHProjection, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[projection] = pointer
	}
	current, err := model.BuildQwen35MTPInput(
		builder, tokenInput, hiddenInput, embeddingNorm, hiddenNorm, projection, r.spec,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	var pastKey, pastValue *tensor.Tensor
	if session.Layer.Key.Shape.Rank != 0 {
		pastKey = builder.Input("qwen35_mtp.past_key", dtype.F32, session.Layer.Key.Shape)
		pastValue = builder.Input("qwen35_mtp.past_value", dtype.F32, session.Layer.Value.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = session.Layer.Key, session.Layer.Value
	}
	block, err := model.BuildQwen35MTPBlockCached(
		builder, current, r.spec, graphWeights, []uint32{session.Position}, pastKey, pastValue,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	outputNormInfo := r.weights.OutputNorm
	if mtp.OutputNorm != nil {
		outputNormInfo = *mtp.OutputNorm
	}
	outputNorm, pointer, err := r.deviceOrHostTensor(ctx, builder, outputNormInfo, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[outputNorm] = pointer
	}
	outputInfo := r.weights.TokenEmbedding
	if r.weights.Output != nil {
		outputInfo = *r.weights.Output
	}
	if mtp.Output != nil {
		outputInfo = *mtp.Output
	}
	output, pointer, err := r.deviceOrHostTensor(ctx, builder, outputInfo, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[output] = pointer
	}
	logits, nextHidden, err := model.BuildQwen35MTPOutputs(builder, block.Output, outputNorm, output, r.spec)
	if err != nil {
		return reference.Value{}, nil, err
	}
	outputs := []*tensor.Tensor{logits, nextHidden, block.Key, block.Value}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, outputs, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, nil, err
	}
	logitValue := results[logits]
	logitValue.Data = r.finalizeLogits(logitValue.Data)
	next := &Qwen35MTPSession{
		TrunkCache:    session.TrunkCache,
		Layer:         LayerCache{Key: results[block.Key], Value: results[block.Value]},
		PendingHidden: results[nextHidden],
		MTPStart:      session.MTPStart,
		Position:      session.Position + 1,
		targetModel:   session.targetModel,
	}
	return logitValue, next, nil
}

func (r *Runner) validateQwen35MTP() error {
	if r.spec.Profile().DraftKind != model.DraftQwen35MTP || r.spec.NextNPredictLayers != 1 ||
		r.weights.Qwen35MTP == nil {
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

func (r *Runner) qwen35MTPLayerInputs(
	ctx context.Context,
	builder *tensor.Builder,
	hostFeeds map[*tensor.Tensor]reference.Value,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	return r.mtpLayerInputs(
		ctx, builder, hostFeeds, r.weights.Qwen35MTP.Layer, fmt.Sprintf("blk.%d.", r.spec.BlockCount),
	)
}
