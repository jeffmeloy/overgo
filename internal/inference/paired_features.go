package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

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
	if r == nil || target == nil || len(tokenIDs) == 0 {
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
	positions := tokenPositions(0, len(tokenIDs))
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
	start := 0
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
	featureWidth := int(features.Shape.Dims[0])
	features = reference.Value{
		Shape: tensor.MustShape(features.Shape.Dims[0], uint64(len(tokenIDs)-start)),
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
	_, rows, valid := fused.MatrixExtents()
	if !valid || rows != len(positions) {
		return nil, errors.New("inference: paired-feature shape is incompatible")
	}
	for index, position := range positions {
		if index > 0 && position != positions[index-1]+1 {
			return nil, errors.New("inference: paired-feature positions are not contiguous")
		}
		if index == 0 && cache != nil && position != cache.Position {
			return nil, errors.New("inference: paired-feature position does not append cache")
		}
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
	if len(positions) > 0 {
		next.Position = positions[len(positions)-1] + 1
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
	if len(tokenIDs) == 0 || len(tokenIDs) != len(positions) || cache == nil || len(cache.Layers) != len(r.weights.Layers) {
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
	if r == nil || cache == nil || draftCount < 1 || draftCount >= int(r.spec.DFlashBlockSize) {
		return reference.Value{}, errors.New("inference: paired-feature draft size is invalid")
	}
	if r.vocab.Mask == tokenizer.NullToken {
		return reference.Value{}, errors.New("inference: paired-feature vocabulary has no mask token")
	}
	ids := make([]tokenizer.TokenID, draftCount+1)
	positions := tokenPositions(cache.Position, draftCount+1)
	ids[0] = last
	for index := range ids {
		if index > 0 {
			ids[index] = r.vocab.Mask
		}
	}
	return r.DecodePairedFeatureBlock(ctx, target, ids, positions, cache)
}
