package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// PairedFeatureSession: synchronized target and injected-feature caches.
type PairedFeatureSession struct {
	Cache        *KVCache
	TargetCache  *KVCache
	TargetTokens []tokenizer.TokenID
	Position     uint32
}

// NewPairedFeatureSession: full-prefix target and feature-cache construction.
func (r *Runner) NewPairedFeatureSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*PairedFeatureSession, error) {
	if r == nil || target == nil || !checked.Nonzero(len(tokenIDs)) {
		return nil, errors.New("inference: paired-feature session inputs are invalid")
	}
	_, targetCache, features, err := target.ForwardCachedExtractLayerInputs(
		ctx, tokenIDs, nil, r.spec.TargetLayers,
	)
	if err != nil {
		return nil, err
	}
	fused, err := r.projectFeatures(ctx, features)
	if err != nil {
		return nil, err
	}
	var origin uint32
	positions := tokenPositions(origin, len(tokenIDs))
	cache, err := r.InjectPairedFeatures(ctx, fused, positions, nil)
	if err != nil {
		return nil, err
	}
	position := uint32(len(tokenIDs))
	if cache.Position != position || effectiveCachePosition(targetCache) != position {
		return nil, errors.New("inference: paired-feature session cache position is inconsistent")
	}
	return &PairedFeatureSession{
		Cache: cache, TargetCache: targetCache,
		TargetTokens: slices.Clone(tokenIDs), Position: position,
	}, nil
}

// SyncPairedFeaturePrefix: recomputes target inputs; injects unsynced suffix.
func (r *Runner) SyncPairedFeaturePrefix(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
) (*KVCache, error) {
	if r == nil || target == nil || r == target || r.path == target.path {
		return nil, errors.New("inference: paired-feature runners are invalid")
	}
	if r.forwardProgram().Session != model.ForwardSessionPairedFeatures || target.spec.EmbeddingLength != r.spec.EmbeddingLength {
		return nil, errors.New("inference: paired-feature target is incompatible")
	}
	var start int
	if cache != nil {
		start = int(cache.Tokens)
		if start > len(tokenIDs) {
			return nil, errors.New("inference: paired-feature cache exceeds target prefix")
		}
		if start == len(tokenIDs) {
			return cache, nil
		}
	}
	features, err := target.ExtractLayerInputs(ctx, tokenIDs, r.spec.TargetLayers)
	if err != nil {
		return nil, err
	}
	featureWidth, _, _ := features.MatrixExtents()
	features = reference.Value{
		Shape: tensor.MustShape(uint64(featureWidth), uint64(len(tokenIDs)-start)),
		Data:  slices.Clone(features.Data[start*featureWidth:]),
	}
	fused, err := r.projectFeatures(ctx, features)
	if err != nil {
		return nil, err
	}
	positions := make([]uint32, len(tokenIDs)-start)
	for index := range positions {
		positions[index] = uint32(start + index)
	}
	return r.InjectPairedFeatures(ctx, fused, positions, cache)
}

// InjectPairedFeatures: appends fused committed-token K/V.
func (r *Runner) InjectPairedFeatures(
	ctx context.Context,
	fused reference.Value,
	positions []uint32,
	cache *KVCache,
) (*KVCache, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	if err := r.lockOpen(); err != nil {
		return nil, err
	}
	defer r.mu.Unlock()
	if r.forwardProgram().Session != model.ForwardSessionPairedFeatures {
		return nil, errors.New("inference: cache injection requires a paired-feature program")
	}
	_, fusedRows, validFused := fused.MatrixExtents()
	if !validFused || !checked.Equal(fusedRows, len(positions)) {
		return nil, errors.New("inference: paired-feature shape is incompatible")
	}
	previousPositions, _ := checked.Init(positions)
	nextPositions, _ := checked.Tail(positions)
	for index, previous := range previousPositions {
		if nextPositions[index] != previous+uint32(tensor.SingletonExtent) {
			return nil, errors.New("inference: paired-feature positions are not contiguous")
		}
	}
	firstPosition, hasPosition := checked.First(positions)
	if hasPosition && cache != nil && firstPosition != cache.Position {
		return nil, errors.New("inference: paired-feature position does not append cache")
	}
	if cache != nil && len(cache.Layers) != len(r.weights.Layers) {
		return nil, errors.New("inference: paired-feature cache layer count is incompatible")
	}
	next := &KVCache{Layers: make([]LayerCache, len(r.weights.Layers))}
	if cache != nil {
		next.Tokens, next.Position = cache.Tokens, cache.Position
	}
	for layerIndex, info := range r.weights.Layers {
		var past *LayerCache
		if cache != nil {
			past = &cache.Layers[layerIndex]
		}
		layer, err := r.injectPairedFeatureLayer(ctx, fused, positions, info, layerIndex, past)
		if err != nil {
			return nil, fmt.Errorf("inference paired-feature injection layer %d: %w", layerIndex, err)
		}
		next.Layers[layerIndex] = layer
	}
	next.Tokens += uint32(len(positions))
	if finalSlice, ok := checked.LastSlice(positions); ok {
		finalPosition, _ := checked.First(finalSlice)
		next.Position = finalPosition + uint32(tensor.SingletonExtent)
	}
	return next, nil
}

func (r *Runner) injectPairedFeatureLayer(
	ctx context.Context,
	fused reference.Value,
	positions []uint32,
	info model.LayerWeights,
	layerIndex int,
	past *LayerCache,
) (LayerCache, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("paired_features.fused", fused)
	graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
	if err != nil {
		return LayerCache{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if past != nil && past.Key.Defined() {
		pastKey = runtime.input("paired_features.past_key", past.Key)
		pastValue = runtime.input("paired_features.past_value", past.Value)
	}
	key, value, err := r.program.Model.CacheProjection().Build(
		runtime.builder, input, graphWeights, positions, pastKey, pastValue,
	)
	if err != nil {
		return LayerCache{}, err
	}
	results, err := runtime.execute(key, value)
	if err != nil {
		return LayerCache{}, err
	}
	return LayerCache{Key: results[key], Value: results[value]}, nil
}

// DecodePairedFeatureBlock: paired-target masked-block logits.
func (r *Runner) DecodePairedFeatureBlock(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
	positions []uint32,
	cache *KVCache,
) (reference.Value, error) {
	if r == nil || target == nil || r == target || r.path == target.path {
		return reference.Value{}, errors.New("inference: paired-feature runners are invalid")
	}
	first, second := r, target
	if first.path > second.path {
		first, second = second, first
	}
	first.mu.Lock()
	second.mu.Lock()
	defer second.mu.Unlock()
	defer first.mu.Unlock()
	if r.closed || target.closed {
		return reference.Value{}, errors.New("inference: runner is closed")
	}
	if r.forwardProgram().Session != model.ForwardSessionPairedFeatures || target.spec.EmbeddingLength != r.spec.EmbeddingLength ||
		target.spec.VocabularySize != r.spec.VocabularySize {
		return reference.Value{}, errors.New("inference: paired-feature target is incompatible")
	}
	if !checked.Nonzero(len(tokenIDs)) || len(tokenIDs) != len(positions) || cache == nil || len(cache.Layers) != len(r.weights.Layers) {
		return reference.Value{}, errors.New("inference: paired-feature block input is incompatible")
	}
	rows, err := target.vocab.TensorIndices(tokenIDs)
	if err != nil {
		return reference.Value{}, err
	}
	activation, err := target.gatherTensor(ctx, target.weights.TokenEmbedding, rows)
	if err != nil {
		return reference.Value{}, err
	}
	for layerIndex, info := range r.weights.Layers {
		activation, _, err = r.runLayerCached(
			ctx, activation, info, layerIndex, positions, nil, &cache.Layers[layerIndex], reference.Value{}, nil, nil, false, nil,
		)
		if err != nil {
			return reference.Value{}, fmt.Errorf("inference paired-feature layer %d: %w", layerIndex, err)
		}
	}
	if !r.hasPreloadedWeights() {
		activation, err = r.runOutputNorm(ctx, activation)
		if err != nil {
			return reference.Value{}, err
		}
	}
	return target.projectAllLogits(ctx, activation)
}

// DraftPairedFeatureBlock: last-token plus MASK-block logits.
func (r *Runner) DraftPairedFeatureBlock(
	ctx context.Context,
	target *Runner,
	last tokenizer.TokenID,
	draftCount int,
	cache *KVCache,
) (reference.Value, error) {
	if r == nil || cache == nil || !checked.PositiveInts(draftCount) || draftCount >= int(r.spec.DFlashBlockSize) {
		return reference.Value{}, errors.New("inference: paired-feature draft size is invalid")
	}
	if r.vocab.Mask == tokenizer.NullToken {
		return reference.Value{}, errors.New("inference: paired-feature vocabulary has no mask token")
	}
	blockLength := draftCount + tensor.SingletonExtent
	ids := make([]tokenizer.TokenID, blockLength)
	positions := tokenPositions(cache.Position, blockLength)
	for index := range ids {
		ids[index] = r.vocab.Mask
	}
	ids[tensor.FirstOffset] = last
	return r.DecodePairedFeatureBlock(ctx, target, ids, positions, cache)
}
