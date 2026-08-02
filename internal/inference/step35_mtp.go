package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Step35MTPSession: trunk snapshot plus per-head draft state.
type Step35MTPSession struct {
	TrunkCache    *KVCache
	Heads         []LayerCache
	PendingHidden reference.Value
	DraftTokens   []tokenizer.TokenID
	DraftHidden   []reference.Value
	MTPStart      uint32
	Position      uint32
	targetModel   [32]byte
}

// HYV3MTPSession: HY-V3 multi-head draft state.
type HYV3MTPSession = Step35MTPSession

// NewStep35MTPSession: trunk prefill plus per-head catch-up.
func (r *Runner) NewStep35MTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*Step35MTPSession, error) {
	if err := r.validateStep35MTP(); err != nil {
		return nil, err
	}
	return r.newMultiHeadMTPSession(ctx, tokenIDs)
}

// NewHYV3MTPSession: HY-V3 trunk and head prefill.
func (r *Runner) NewHYV3MTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*HYV3MTPSession, error) {
	if err := r.validateHYV3MTP(); err != nil {
		return nil, err
	}
	return r.newMultiHeadMTPSession(ctx, tokenIDs)
}

func (r *Runner) newMultiHeadMTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*Step35MTPSession, error) {
	if r == nil || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Step3.5 MTP inputs are invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: Step3.5 MTP runner is unavailable")
	}
	if err := r.validateMultiHeadMTP(); err != nil {
		return nil, err
	}
	var hidden reference.Value
	var cache *KVCache
	var err error
	if r.spec.Architecture == "step35" {
		hidden, cache, err = r.forwardCachedPreOutputNormLocked(ctx, tokenIDs, nil)
	} else {
		hidden, cache, err = r.forwardCachedLocked(ctx, tokenIDs, nil)
	}
	if err != nil {
		return nil, err
	}
	width := int(r.spec.EmbeddingLength)
	shifted := reference.Value{
		Shape: tensor.MustShape(uint64(width), uint64(len(tokenIDs))),
		Data:  make([]float32, width*len(tokenIDs)),
	}
	if len(tokenIDs) > 1 {
		copy(shifted.Data[width:], hidden.Data[:len(hidden.Data)-width])
	}
	positions := make([]uint32, len(tokenIDs))
	for index := range positions {
		positions[index] = uint32(index)
	}
	heads := make([]LayerCache, len(r.multiHeadMTPWeights()))
	for offset := range heads {
		_, _, headCache, runErr := r.runMultiHeadMTPHeadLocked(
			ctx, tokenIDs, shifted, positions, nil, uint32(offset),
		)
		if runErr != nil {
			return nil, fmt.Errorf("inference: Step3.5 MTP head %d prefill: %w", offset, runErr)
		}
		heads[offset] = headCache
	}
	targetModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	last := reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), hidden.Data[len(hidden.Data)-width:]...),
	}
	position := effectiveCachePosition(cache)
	return &Step35MTPSession{
		TrunkCache: cache, Heads: heads, PendingHidden: last,
		MTPStart: position, Position: position, targetModel: targetModel,
	}, nil
}

// AdvanceStep35MTP: next trained head over growing draft prefix.
func (r *Runner) AdvanceStep35MTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *Step35MTPSession,
) (reference.Value, *Step35MTPSession, error) {
	if err := r.validateStep35MTP(); err != nil {
		return reference.Value{}, nil, err
	}
	return r.advanceMultiHeadMTP(ctx, tokenID, session)
}

// AdvanceHYV3MTP: next HY-V3 trained head.
func (r *Runner) AdvanceHYV3MTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *HYV3MTPSession,
) (reference.Value, *HYV3MTPSession, error) {
	if err := r.validateHYV3MTP(); err != nil {
		return reference.Value{}, nil, err
	}
	return r.advanceMultiHeadMTP(ctx, tokenID, session)
}

func (r *Runner) advanceMultiHeadMTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *Step35MTPSession,
) (reference.Value, *Step35MTPSession, error) {
	if r == nil || session == nil {
		return reference.Value{}, nil, errors.New("inference: Step3.5 MTP session is invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: Step3.5 MTP runner is unavailable")
	}
	if err := r.validateMultiHeadMTP(); err != nil {
		return reference.Value{}, nil, err
	}
	if err := r.validateStep35MTPSession(session); err != nil {
		return reference.Value{}, nil, err
	}
	if tokenID < 0 || int(tokenID) >= r.vocab.Len() {
		return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	offset := len(session.DraftTokens)
	if offset >= len(r.multiHeadMTPWeights()) {
		return reference.Value{}, nil, errors.New("inference: Step3.5 MTP head chain is exhausted")
	}
	tokens := append(append([]tokenizer.TokenID(nil), session.DraftTokens...), tokenID)
	width := int(r.spec.EmbeddingLength)
	hidden := reference.Value{
		Shape: tensor.MustShape(uint64(width), uint64(len(tokens))),
		Data:  make([]float32, width*len(tokens)),
	}
	copy(hidden.Data, session.PendingHidden.Data)
	for index, row := range session.DraftHidden {
		copy(hidden.Data[(index+1)*width:], row.Data)
	}
	positions := make([]uint32, len(tokens))
	for index := range positions {
		positions[index] = session.MTPStart + uint32(index)
	}
	logits, nextHidden, headCache, err := r.runMultiHeadMTPHeadLocked(
		ctx, tokens, hidden, positions, &session.Heads[offset], uint32(offset),
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	lastLogits, err := lastValueColumn(logits)
	if err != nil {
		return reference.Value{}, nil, err
	}
	lastHidden, err := lastValueColumn(nextHidden)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &Step35MTPSession{
		TrunkCache: session.TrunkCache,
		Heads:      append([]LayerCache(nil), session.Heads...),
		PendingHidden: reference.Value{
			Shape: session.PendingHidden.Shape,
			Data:  append([]float32(nil), session.PendingHidden.Data...),
		},
		DraftTokens: append(append([]tokenizer.TokenID(nil), session.DraftTokens...), tokenID),
		DraftHidden: append(append([]reference.Value(nil), session.DraftHidden...), lastHidden),
		MTPStart:    session.MTPStart,
		Position:    session.MTPStart + uint32(len(tokens)),
		targetModel: session.targetModel,
	}
	next.Heads[offset] = headCache
	lastLogits.Data = r.finalizeLogits(lastLogits.Data)
	return lastLogits, next, nil
}

func (r *Runner) runMultiHeadMTPHeadLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	hidden reference.Value,
	positions []uint32,
	past *LayerCache,
	offset uint32,
) (reference.Value, reference.Value, LayerCache, error) {
	mtpWeights := r.multiHeadMTPWeights()
	if offset >= uint32(len(mtpWeights)) || len(tokenIDs) == 0 || len(positions) != len(tokenIDs) {
		return reference.Value{}, reference.Value{}, LayerCache{}, errors.New("inference: Step3.5 MTP head inputs are invalid")
	}
	rows := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return reference.Value{}, reference.Value{}, LayerCache{}, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
	}
	mtp := mtpWeights[offset]
	embeddingInfo := r.weights.TokenEmbedding
	if mtp.TokenEmbedding != nil {
		embeddingInfo = *mtp.TokenEmbedding
	}
	tokenEmbedding, err := r.loadRows(ctx, embeddingInfo, rows)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	if !hidden.Shape.Equal(tokenEmbedding.Shape) {
		return reference.Value{}, reference.Value{}, LayerCache{}, errors.New("inference: Step3.5 MTP hidden shape is incompatible")
	}
	builder := r.newGraphBuilder()
	tokenInput := builder.Input("step35_mtp.token", dtype.F32, tokenEmbedding.Shape)
	hiddenInput := builder.Input("step35_mtp.hidden", dtype.F32, hidden.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{tokenInput: tokenEmbedding, hiddenInput: hidden}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	graphWeights, layerFeeds, err := r.multiHeadMTPLayerInputs(ctx, builder, hostFeeds, offset)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	for node, pointer := range layerFeeds {
		deviceFeeds[node] = pointer
	}
	load := func(info gguf.TensorInfo) (*tensor.Tensor, error) {
		node, pointer, loadErr := r.deviceOrHostTensor(ctx, builder, info, hostFeeds)
		if pointer != 0 {
			deviceFeeds[node] = pointer
		}
		return node, loadErr
	}
	embeddingNorm, err := load(mtp.EmbeddingNorm)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	hiddenNorm, err := load(mtp.HiddenNorm)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	projection, err := load(mtp.EHProjection)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	var current *tensor.Tensor
	if r.spec.Architecture == "step35" {
		current, err = model.BuildStep35MTPInput(
			builder, tokenInput, hiddenInput, embeddingNorm, hiddenNorm, projection, r.spec, offset,
		)
	} else {
		current, err = model.BuildHYV3MTPInput(
			builder, tokenInput, hiddenInput, embeddingNorm, hiddenNorm, projection, r.spec, offset,
		)
	}
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if past != nil && past.Key.Shape.Rank != 0 {
		pastKey = builder.Input("step35_mtp.past_key", dtype.F32, past.Key.Shape)
		pastValue = builder.Input("step35_mtp.past_value", dtype.F32, past.Value.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = past.Key, past.Value
	}
	var block model.DenseBlockResult
	if r.spec.Architecture == "step35" {
		block, err = model.BuildStep35MTPBlockCached(
			builder, current, r.spec, graphWeights, positions, pastKey, pastValue, offset,
		)
	} else {
		block, err = model.BuildHYV3MTPBlockCached(
			builder, current, r.spec, graphWeights, positions, pastKey, pastValue, offset,
		)
	}
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	outputNormInfo := r.weights.OutputNorm
	if mtp.OutputNorm != nil {
		outputNormInfo = *mtp.OutputNorm
	}
	outputNorm, err := load(outputNormInfo)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	outputInfo := r.weights.TokenEmbedding
	if r.weights.Output != nil {
		outputInfo = *r.weights.Output
	}
	if mtp.Output != nil {
		outputInfo = *mtp.Output
	}
	output, err := load(outputInfo)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	var logits, nextHidden *tensor.Tensor
	if r.spec.Architecture == "step35" {
		logits, nextHidden, err = model.BuildStep35MTPOutputs(
			builder, block.Output, outputNorm, output, r.spec, offset,
		)
	} else {
		logits, nextHidden, err = model.BuildHYV3MTPOutputs(
			builder, block.Output, outputNorm, output, r.spec, offset,
		)
	}
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	outputs := []*tensor.Tensor{logits, nextHidden, block.Key, block.Value}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, outputs, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	return results[logits], results[nextHidden], LayerCache{
		Key: results[block.Key], Value: results[block.Value],
	}, nil
}

func (r *Runner) multiHeadMTPLayerInputs(
	ctx context.Context,
	builder *tensor.Builder,
	hostFeeds map[*tensor.Tensor]reference.Value,
	offset uint32,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	return r.mtpLayerInputs(
		ctx, builder, hostFeeds, r.multiHeadMTPWeights()[offset].Layer,
		fmt.Sprintf("blk.%d.", r.spec.BlockCount+offset),
	)
}

func (r *Runner) validateStep35MTP() error {
	if r == nil || r.spec.Profile().DraftKind != model.DraftStep35MTP || r.spec.NextNPredictLayers == 0 ||
		len(r.weights.Step35MTP) != int(r.spec.NextNPredictLayers) {
		return errors.New("inference: model has no supported Step3.5 MTP heads")
	}
	return nil
}

func (r *Runner) validateHYV3MTP() error {
	if r == nil || r.spec.Profile().DraftKind != model.DraftHYV3MTP || r.spec.NextNPredictLayers == 0 ||
		len(r.weights.HYV3MTP) != int(r.spec.NextNPredictLayers) {
		return errors.New("inference: model has no supported HY-V3 MTP heads")
	}
	return nil
}

func (r *Runner) validateMultiHeadMTP() error {
	if r == nil {
		return errors.New("inference: MTP runner is nil")
	}
	if r.spec.Architecture == "hy_v3" {
		return r.validateHYV3MTP()
	}
	return r.validateStep35MTP()
}

func (r *Runner) multiHeadMTPWeights() []model.Step35MTPWeights {
	if r != nil && r.spec.Architecture == "hy_v3" {
		return r.weights.HYV3MTP
	}
	if r == nil {
		return nil
	}
	return r.weights.Step35MTP
}

func (r *Runner) validateStep35MTPSession(session *Step35MTPSession) error {
	if session == nil || session.TrunkCache == nil || len(session.Heads) != len(r.multiHeadMTPWeights()) ||
		len(session.DraftTokens) != len(session.DraftHidden) || len(session.DraftTokens) > len(session.Heads) ||
		session.Position != session.MTPStart+uint32(len(session.DraftTokens)) || session.Position == math.MaxUint32 {
		return errors.New("inference: Step3.5 MTP session state is incompatible")
	}
	if err := r.validateCache(session.TrunkCache); err != nil {
		return fmt.Errorf("inference: Step3.5 MTP trunk cache: %w", err)
	}
	if session.MTPStart != effectiveCachePosition(session.TrunkCache) ||
		session.PendingHidden.Shape.Rank != 2 ||
		session.PendingHidden.Shape.Dims[0] != uint64(r.spec.EmbeddingLength) ||
		session.PendingHidden.Shape.Dims[1] != 1 {
		return errors.New("inference: Step3.5 MTP session state is incompatible")
	}
	for _, hidden := range session.DraftHidden {
		if !hidden.Shape.Equal(session.PendingHidden.Shape) {
			return errors.New("inference: Step3.5 MTP draft hidden state is incompatible")
		}
	}
	for offset, layer := range session.Heads {
		tokens := session.MTPStart
		if offset < len(session.DraftTokens) {
			tokens += uint32(offset + 1)
		}
		block := r.spec.BlockCount + uint32(offset)
		keyShape := tensor.MustShape(
			uint64(r.spec.LayerKeyLength(block)),
			uint64(r.spec.LayerKVHeadCount(block)),
			uint64(tokens),
		)
		valueShape := tensor.MustShape(
			uint64(r.spec.LayerValueLength(block)),
			uint64(r.spec.LayerKVHeadCount(block)),
			uint64(tokens),
		)
		if !layer.Key.Shape.Equal(keyShape) || !layer.Value.Shape.Equal(valueShape) ||
			len(layer.Key.Data) == 0 || len(layer.Value.Data) == 0 {
			return fmt.Errorf("inference: Step3.5 MTP head %d cache is incompatible", offset)
		}
	}
	return nil
}

func lastValueColumn(value reference.Value) (reference.Value, error) {
	if value.Shape.Rank != 2 || value.Shape.Dims[0] == 0 || value.Shape.Dims[1] == 0 {
		return reference.Value{}, errors.New("inference: output has no final column")
	}
	width := int(value.Shape.Dims[0])
	if len(value.Data) != width*int(value.Shape.Dims[1]) {
		return reference.Value{}, errors.New("inference: output storage is invalid")
	}
	return reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), value.Data[len(value.Data)-width:]...),
	}, nil
}
