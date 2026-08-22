package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// MultiHeadMTPSession: trunk snapshot plus per-head draft state.
type MultiHeadMTPSession struct {
	TrunkCache    *KVCache
	Heads         []LayerCache
	PendingHidden reference.Value
	DraftTokens   []tokenizer.TokenID
	DraftHidden   []reference.Value
	MTPStart      uint32
	Position      uint32
	targetModel   [32]byte
}

// NewMultiHeadMTPSession compiles trunk prefill plus per-head catch-up.
func (r *Runner) NewMultiHeadMTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*MultiHeadMTPSession, error) {
	if r == nil || len(tokenIDs) == 0 {
		return nil, errors.New("inference: multi-head MTP inputs are invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: multi-head MTP runner is unavailable")
	}
	plan, err := r.multiHeadMTP()
	if err != nil {
		return nil, err
	}
	var hidden reference.Value
	var cache *KVCache
	if plan.CarryRawHidden {
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
	positions := tokenPositions(0, len(tokenIDs))
	heads := make([]LayerCache, int(plan.Heads))
	for offset := range heads {
		_, _, headCache, runErr := r.runMultiHeadMTPHeadLocked(
			ctx, tokenIDs, shifted, positions, nil, uint32(offset),
		)
		if runErr != nil {
			return nil, fmt.Errorf("inference: multi-head MTP head %d prefill: %w", offset, runErr)
		}
		heads[offset] = headCache
	}
	targetModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	last := hidden.LastRowView().Clone()
	position := effectiveCachePosition(cache)
	return &MultiHeadMTPSession{
		TrunkCache: cache, Heads: heads, PendingHidden: last,
		MTPStart: position, Position: position, targetModel: targetModel,
	}, nil
}

// AdvanceMultiHeadMTP executes the next compiled trained head.
func (r *Runner) AdvanceMultiHeadMTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *MultiHeadMTPSession,
) (reference.Value, *MultiHeadMTPSession, error) {
	if r == nil || session == nil {
		return reference.Value{}, nil, errors.New("inference: multi-head MTP session is invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: multi-head MTP runner is unavailable")
	}
	plan, err := r.multiHeadMTP()
	if err != nil {
		return reference.Value{}, nil, err
	}
	if err := r.validateMultiHeadMTPSession(session); err != nil {
		return reference.Value{}, nil, err
	}
	if tokenID < 0 || int(tokenID) >= r.vocab.Len() {
		return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	offset := len(session.DraftTokens)
	if offset >= int(plan.Heads) {
		return reference.Value{}, nil, errors.New("inference: multi-head MTP head chain is exhausted")
	}
	tokens := append(slices.Clone(session.DraftTokens), tokenID)
	width := int(r.spec.EmbeddingLength)
	hidden := reference.Value{
		Shape: tensor.MustShape(uint64(width), uint64(len(tokens))),
		Data:  make([]float32, width*len(tokens)),
	}
	copy(hidden.Data, session.PendingHidden.Data)
	for index, row := range session.DraftHidden {
		copy(hidden.Data[(index+1)*width:], row.Data)
	}
	positions := tokenPositions(session.MTPStart, len(tokens))
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
	next := &MultiHeadMTPSession{
		TrunkCache: session.TrunkCache,
		Heads:      slices.Clone(session.Heads),
		PendingHidden: reference.Value{
			Shape: session.PendingHidden.Shape,
			Data:  slices.Clone(session.PendingHidden.Data),
		},
		DraftTokens: append(slices.Clone(session.DraftTokens), tokenID),
		DraftHidden: append(slices.Clone(session.DraftHidden), lastHidden),
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
	plan, err := r.multiHeadMTP()
	if err != nil || !plan.HasHead(offset) || len(tokenIDs) == 0 || len(positions) != len(tokenIDs) {
		return reference.Value{}, reference.Value{}, LayerCache{}, errors.New("inference: multi-head MTP head inputs are invalid")
	}
	rows, err := r.vocab.TensorIndices(tokenIDs)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	mtp, ok := r.weights.DraftCatalog(plan.Kind, offset)
	if !ok {
		return reference.Value{}, reference.Value{}, LayerCache{}, errors.New("inference: compiled MTP catalog is unavailable")
	}
	embeddingInfo := r.weights.TokenEmbedding
	if mtp.TokenEmbedding != nil {
		embeddingInfo = *mtp.TokenEmbedding
	}
	tokenEmbedding, err := r.gatherTensor(ctx, embeddingInfo, rows)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	if !hidden.Shape.Equal(tokenEmbedding.Shape) {
		return reference.Value{}, reference.Value{}, LayerCache{}, errors.New("inference: multi-head MTP hidden shape is incompatible")
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	tokenInput := runtime.input("multi_head_mtp.token", tokenEmbedding)
	hiddenInput := runtime.input("multi_head_mtp.hidden", hidden)
	graphWeights, err := runtime.layer(
		mtp.Layer, fmt.Sprintf("blk.%d.", plan.Block(r.spec.BlockCount, offset)),
	)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	embeddingNorm, err := runtime.weight(mtp.EmbeddingNorm)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	hiddenNorm, err := runtime.weight(mtp.HiddenNorm)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	projection, err := runtime.weight(mtp.EHProjection)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	draftProgram, err := r.draftLayerProgram(offset)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	current, err := draftProgram.BuildDraftInput(
		runtime.builder, tokenInput, hiddenInput, embeddingNorm, hiddenNorm, projection,
	)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if past != nil && past.Key.Defined() {
		pastKey = runtime.input("multi_head_mtp.past_key", past.Key)
		pastValue = runtime.input("multi_head_mtp.past_value", past.Value)
	}
	layerPlan := draftProgram.Layer()
	block, err := draftProgram.Build(model.CachedBlockContext{
		Builder: runtime.builder, Input: current, Positions: positions,
		PastKey: pastKey, PastValue: pastValue, Layer: layerPlan.Layer,
	}, graphWeights)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	outputNormInfo := r.outputNormTensorFor(mtp.OutputNorm)
	outputNorm, err := runtime.weight(outputNormInfo)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	outputInfo := r.outputTensorFor(mtp.Output)
	output, err := runtime.weight(outputInfo)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	logits, nextHidden, err := draftProgram.BuildDraftOutputs(
		runtime.builder, block.Output, outputNorm, output,
	)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	results, err := runtime.execute(logits, nextHidden, block.Key, block.Value)
	if err != nil {
		return reference.Value{}, reference.Value{}, LayerCache{}, err
	}
	return results[logits], results[nextHidden], LayerCache{
		Key: results[block.Key], Value: results[block.Value],
	}, nil
}

func (r *Runner) multiHeadMTP() (model.DraftPlan, error) {
	if r == nil {
		return model.DraftPlan{}, errors.New("inference: MTP runner is nil")
	}
	plan, ok := r.lookupMultiHeadMTP()
	if !ok {
		return model.DraftPlan{}, errors.New("inference: model has no complete multi-head MTP program")
	}
	return plan, nil
}

func (r *Runner) lookupMultiHeadMTP() (model.DraftPlan, bool) {
	if r == nil {
		return model.DraftPlan{}, false
	}
	plan := r.program.Model.Draft()
	if plan.Session != model.DraftSessionMulti || !plan.AppendedBlocks || !plan.SessionEligible() {
		return model.DraftPlan{}, false
	}
	catalog, ok := r.weights.DraftCatalog(plan.Kind, plan.Heads-1)
	if !ok || catalog.Kind != plan.Kind {
		return model.DraftPlan{}, false
	}
	return plan, true
}

func (r *Runner) validateMultiHeadMTPSession(session *MultiHeadMTPSession) error {
	plan, err := r.multiHeadMTP()
	if err != nil {
		return err
	}
	if session == nil || session.TrunkCache == nil || len(session.Heads) != int(plan.Heads) ||
		len(session.DraftTokens) != len(session.DraftHidden) || len(session.DraftTokens) > len(session.Heads) ||
		session.Position != session.MTPStart+uint32(len(session.DraftTokens)) || session.Position == math.MaxUint32 {
		return errors.New("inference: multi-head MTP session state is incompatible")
	}
	if err := r.validateCache(session.TrunkCache); err != nil {
		return fmt.Errorf("inference: multi-head MTP trunk cache: %w", err)
	}
	pendingRows, pendingValid := r.spec.SequenceRows(session.PendingHidden)
	if session.MTPStart != effectiveCachePosition(session.TrunkCache) ||
		!pendingValid || pendingRows != tensor.SingletonExtent {
		return errors.New("inference: multi-head MTP session state is incompatible")
	}
	for _, hidden := range session.DraftHidden {
		if !hidden.Shape.Equal(session.PendingHidden.Shape) {
			return errors.New("inference: multi-head MTP draft hidden state is incompatible")
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
			return fmt.Errorf("inference: multi-head MTP head %d cache is incompatible", offset)
		}
	}
	return nil
}

func lastValueColumn(value reference.Value) (reference.Value, error) {
	if value.Shape.Rank != 2 || value.Shape.Dims[0] == 0 || value.Shape.Dims[1] == 0 {
		return reference.Value{}, errors.New("inference: output has no final column")
	}
	result, err := value.TailRows(1)
	if err != nil {
		return reference.Value{}, fmt.Errorf("inference: output storage is invalid: %w", err)
	}
	return result, nil
}
