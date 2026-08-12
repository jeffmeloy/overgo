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

// DFlashSession: synchronized target and injected-feature caches.
type DFlashSession struct {
	Cache        *KVCache
	TargetCache  *KVCache
	TargetTokens []tokenizer.TokenID
	Position     uint32
}

// NewDFlashSession: full-prefix target and feature-cache construction.
func (r *Runner) NewDFlashSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*DFlashSession, error) {
	if r == nil || target == nil || len(tokenIDs) == 0 {
		return nil, errors.New("inference: DFlash session inputs are invalid")
	}
	_, targetCache, features, err := target.ForwardCachedExtractLayerInputs(
		ctx, tokenIDs, nil, r.spec.TargetLayers,
	)
	if err != nil {
		return nil, err
	}
	fused, err := r.FuseDFlashFeatures(ctx, features)
	if err != nil {
		return nil, err
	}
	positions := tokenPositions(0, len(tokenIDs))
	cache, err := r.InjectDFlashFeatures(ctx, fused, positions, nil)
	if err != nil {
		return nil, err
	}
	position := uint32(len(tokenIDs))
	if cache.Position != position || effectiveCachePosition(targetCache) != position {
		return nil, errors.New("inference: DFlash session cache position is inconsistent")
	}
	return &DFlashSession{
		Cache: cache, TargetCache: targetCache,
		TargetTokens: slices.Clone(tokenIDs), Position: position,
	}, nil
}

// ExtractLayerInputs: full-sequence pre-layer hidden rows.
func (r *Runner) ExtractLayerInputs(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	layerIDs []int32,
) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: runner is closed")
	}
	if len(tokenIDs) == 0 || len(layerIDs) == 0 || len(tokenIDs) > int(r.spec.ContextLength) {
		return reference.Value{}, errors.New("inference: layer extraction input is invalid")
	}
	requested := make(map[int32]struct{}, len(layerIDs))
	for _, layer := range layerIDs {
		if layer < 0 || int(layer) >= len(r.weights.Layers) {
			return reference.Value{}, fmt.Errorf("inference: extraction layer %d is out of range", layer)
		}
		requested[layer] = struct{}{}
	}
	rows, err := r.tokenRows(tokenIDs)
	if err != nil {
		return reference.Value{}, err
	}
	positions := tokenPositions(0, len(tokenIDs))
	activation, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, err
	}
	activation, err = r.addTokenTypeEmbedding(ctx, activation)
	if err != nil {
		return reference.Value{}, err
	}
	activation, err = r.addPositionEmbeddings(ctx, activation, positions)
	if err != nil {
		return reference.Value{}, err
	}
	if scale := r.spec.InputEmbeddingScale(); scale != 1 {
		for index := range activation.Data {
			activation.Data[index] *= scale
		}
	}
	activation, err = r.applyTokenEmbeddingNorm(ctx, activation)
	if err != nil {
		return reference.Value{}, err
	}
	extracted := make(map[int32]reference.Value, len(layerIDs))
	for layerIndex, info := range r.weights.Layers {
		if _, ok := requested[int32(layerIndex)]; ok {
			extracted[int32(layerIndex)] = activation
		}
		activation, _, err = r.runLayerCached(
			ctx, activation, info, layerIndex, positions, nil, nil, reference.Value{}, nil, nil, false, nil,
		)
		if err != nil {
			return reference.Value{}, fmt.Errorf("inference extraction layer %d: %w", layerIndex, err)
		}
	}
	width, tokens := int(r.spec.EmbeddingLength), len(tokenIDs)
	result := reference.Value{
		Shape: tensor.MustShape(uint64(width*len(layerIDs)), uint64(tokens)),
		Data:  make([]float32, width*len(layerIDs)*tokens),
	}
	for token := 0; token < tokens; token++ {
		for order, layer := range layerIDs {
			value := extracted[layer]
			source := value.Data[token*width : (token+1)*width]
			destination := (token*len(layerIDs) + order) * width
			copy(result.Data[destination:destination+width], source)
		}
	}
	return result, nil
}

// SyncDFlashPrefix: recomputes target inputs; injects unsynced suffix.
func (r *Runner) SyncDFlashPrefix(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
) (*KVCache, error) {
	if r == nil || target == nil || r == target || r.path == target.path {
		return nil, errors.New("inference: DFlash and target runners are invalid")
	}
	if r.forwardProgram().Session != model.ForwardSessionPairedFeatures || target.spec.EmbeddingLength != r.spec.EmbeddingLength {
		return nil, errors.New("inference: DFlash target model is incompatible")
	}
	start := 0
	if cache != nil {
		start = int(cache.Tokens)
		if start > len(tokenIDs) {
			return nil, errors.New("inference: DFlash cache exceeds target prefix")
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
	fused, err := r.FuseDFlashFeatures(ctx, features)
	if err != nil {
		return nil, err
	}
	positions := make([]uint32, len(tokenIDs)-start)
	for index := range positions {
		positions[index] = uint32(start + index)
	}
	return r.InjectDFlashFeatures(ctx, fused, positions, cache)
}

// FuseDFlashFeatures: projects concatenated target-layer inputs.
func (r *Runner) FuseDFlashFeatures(ctx context.Context, features reference.Value) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: runner is closed")
	}
	if r.forwardProgram().Session != model.ForwardSessionPairedFeatures || r.weights.FeatureProjection == nil || r.weights.EncoderOutputNorm == nil {
		return reference.Value{}, errors.New("inference: DFlash feature encoder is unavailable")
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("dflash.features", features)
	projection, err := runtime.weight(*r.weights.FeatureProjection)
	if err != nil {
		return reference.Value{}, err
	}
	projectionNorm, err := runtime.weight(*r.weights.EncoderOutputNorm)
	if err != nil {
		return reference.Value{}, err
	}
	result, err := r.program.Model.Projection(model.ProjectionFeature).Build(
		runtime.builder,
		model.ProjectionOperands{Input: input, Primary: projection, Normalization: projectionNorm},
	)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(result.Primary)
	if err != nil {
		return reference.Value{}, err
	}
	return results[result.Primary], nil
}

// InjectDFlashFeatures: appends fused committed-token K/V.
func (r *Runner) InjectDFlashFeatures(
	ctx context.Context,
	fused reference.Value,
	positions []uint32,
	cache *KVCache,
) (*KVCache, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: runner is closed")
	}
	if r.forwardProgram().Session != model.ForwardSessionPairedFeatures {
		return nil, errors.New("inference: cache injection requires DFlash architecture")
	}
	if fused.Shape.Rank != 2 || fused.Shape.Dims[1] != uint64(len(positions)) {
		return nil, errors.New("inference: DFlash fused feature shape is incompatible")
	}
	for index, position := range positions {
		if index > 0 && position != positions[index-1]+1 {
			return nil, errors.New("inference: DFlash injection positions are not contiguous")
		}
		if index == 0 && cache != nil && position != cache.Position {
			return nil, errors.New("inference: DFlash injection position does not append cache")
		}
	}
	if cache != nil && len(cache.Layers) != len(r.weights.Layers) {
		return nil, errors.New("inference: DFlash cache layer count is incompatible")
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
		layer, err := r.injectDFlashLayer(ctx, fused, positions, info, layerIndex, past)
		if err != nil {
			return nil, fmt.Errorf("inference DFlash injection layer %d: %w", layerIndex, err)
		}
		next.Layers[layerIndex] = layer
	}
	next.Tokens += uint32(len(positions))
	if len(positions) > 0 {
		next.Position = positions[len(positions)-1] + 1
	}
	return next, nil
}

func (r *Runner) injectDFlashLayer(
	ctx context.Context,
	fused reference.Value,
	positions []uint32,
	info model.LayerWeights,
	layerIndex int,
	past *LayerCache,
) (LayerCache, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("dflash.fused", fused)
	graphWeights, err := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
	if err != nil {
		return LayerCache{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if past != nil && past.Key.Shape.Rank != 0 {
		pastKey = runtime.input("dflash.past_key", past.Key)
		pastValue = runtime.input("dflash.past_value", past.Value)
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

// DecodeDFlashNoiseBlock: paired-target masked-block logits.
func (r *Runner) DecodeDFlashNoiseBlock(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
	positions []uint32,
	cache *KVCache,
) (reference.Value, error) {
	if r == nil || target == nil || r == target || r.path == target.path {
		return reference.Value{}, errors.New("inference: DFlash and target runners are invalid")
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
		return reference.Value{}, errors.New("inference: DFlash target model is incompatible")
	}
	if len(tokenIDs) == 0 || len(tokenIDs) != len(positions) || cache == nil || len(cache.Layers) != len(r.weights.Layers) {
		return reference.Value{}, errors.New("inference: DFlash noise block input is incompatible")
	}
	rows, err := target.tokenRows(tokenIDs)
	if err != nil {
		return reference.Value{}, err
	}
	activation, err := target.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, err
	}
	for layerIndex, info := range r.weights.Layers {
		activation, _, err = r.runLayerCached(
			ctx, activation, info, layerIndex, positions, nil, &cache.Layers[layerIndex], reference.Value{}, nil, nil, false, nil,
		)
		if err != nil {
			return reference.Value{}, fmt.Errorf("inference DFlash noise layer %d: %w", layerIndex, err)
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

// DraftDFlashBlock: last-token plus MASK-block logits.
func (r *Runner) DraftDFlashBlock(
	ctx context.Context,
	target *Runner,
	last tokenizer.TokenID,
	draftCount int,
	cache *KVCache,
) (reference.Value, error) {
	if r == nil || cache == nil || draftCount < 1 || draftCount >= int(r.spec.DFlashBlockSize) {
		return reference.Value{}, errors.New("inference: DFlash draft size is invalid")
	}
	if r.vocab.Mask == tokenizer.NullToken {
		return reference.Value{}, errors.New("inference: DFlash vocabulary has no mask token")
	}
	ids := make([]tokenizer.TokenID, draftCount+1)
	positions := tokenPositions(cache.Position, draftCount+1)
	ids[0] = last
	for index := range ids {
		if index > 0 {
			ids[index] = r.vocab.Mask
		}
	}
	return r.DecodeDFlashNoiseBlock(ctx, target, ids, positions, cache)
}
