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
	rows := make([]uint32, len(tokenIDs))
	positions := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return reference.Value{}, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index], positions[index] = uint32(id), uint32(index)
	}
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
			ctx, activation, info, layerIndex, positions, nil, nil, reference.Value{}, nil,
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

// PrimeDFlash: extracts, fuses, and injects one committed prefix.
func (r *Runner) PrimeDFlash(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*KVCache, error) {
	return r.SyncDFlashPrefix(ctx, target, tokenIDs, nil)
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
	if r.spec.Architecture != "dflash" || target.spec.EmbeddingLength != r.spec.EmbeddingLength {
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
		Data:  append([]float32(nil), features.Data[start*featureWidth:]...),
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
	if r.spec.Architecture != "dflash" || r.weights.FeatureProjection == nil || r.weights.EncoderOutputNorm == nil {
		return reference.Value{}, errors.New("inference: DFlash feature encoder is unavailable")
	}
	builder := r.newGraphBuilder()
	input := builder.Input("dflash.features", dtype.F32, features.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: features}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	projection, projectionNorm, err := r.dflashEncoderInputs(ctx, builder, hostFeeds, deviceFeeds)
	if err != nil {
		return reference.Value{}, err
	}
	output, err := model.BuildDFlashFeatureEncoder(builder, input, projection, projectionNorm, r.spec)
	if err != nil {
		return reference.Value{}, err
	}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{output}, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, []*tensor.Tensor{output}, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) dflashEncoderInputs(
	ctx context.Context,
	builder *tensor.Builder,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (*tensor.Tensor, *tensor.Tensor, error) {
	if r.hasPreloadedWeights() {
		projection, projectionPointer, err := r.deviceInput(builder, *r.weights.FeatureProjection)
		if err != nil {
			return nil, nil, err
		}
		norm, normPointer, err := r.deviceInput(builder, *r.weights.EncoderOutputNorm)
		if err != nil {
			return nil, nil, err
		}
		deviceFeeds[projection], deviceFeeds[norm] = projectionPointer, normPointer
		return projection, norm, nil
	}
	projectionValue, err := model.LoadHostTensor(ctx, r.file, *r.weights.FeatureProjection)
	if err != nil {
		return nil, nil, err
	}
	normValue, err := model.LoadHostTensor(ctx, r.file, *r.weights.EncoderOutputNorm)
	if err != nil {
		return nil, nil, err
	}
	projection := builder.Input("fc.weight", dtype.F32, projectionValue.Shape)
	norm := builder.Input("enc.output_norm.weight", dtype.F32, normValue.Shape)
	hostFeeds[projection], hostFeeds[norm] = projectionValue, normValue
	return projection, norm, nil
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
	if r.spec.Architecture != "dflash" {
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
	builder := r.newGraphBuilder()
	input := builder.Input("dflash.fused", dtype.F32, fused.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: fused}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var graphWeights model.LayerGraphWeights
	if r.hasPreloadedWeights() {
		var err error
		graphWeights, deviceFeeds, err = r.layerDeviceInputs(builder, info)
		if err != nil {
			return LayerCache{}, err
		}
	} else {
		hostLayer, err := model.LoadHostLayer(ctx, r.file, info)
		if err != nil {
			return LayerCache{}, err
		}
		var layerFeeds map[*tensor.Tensor]reference.Value
		graphWeights, layerFeeds, err = hostLayer.GraphInputs(builder, fmt.Sprintf("blk.%d.", layerIndex))
		if err != nil {
			return LayerCache{}, err
		}
		for node, value := range layerFeeds {
			hostFeeds[node] = value
		}
	}
	var pastKey, pastValue *tensor.Tensor
	if past != nil && past.Key.Shape.Rank != 0 {
		pastKey = builder.Input("dflash.past_key", dtype.F32, past.Key.Shape)
		pastValue = builder.Input("dflash.past_value", dtype.F32, past.Value.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = past.Key, past.Value
	}
	key, value, err := model.BuildDFlashCacheInjection(
		builder, input, r.spec, graphWeights, positions, pastKey, pastValue,
	)
	if err != nil {
		return LayerCache{}, err
	}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{key, value}, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, []*tensor.Tensor{key, value}, hostFeeds)
	}
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
	if r.spec.Architecture != "dflash" || target.spec.EmbeddingLength != r.spec.EmbeddingLength ||
		target.spec.VocabularySize != r.spec.VocabularySize {
		return reference.Value{}, errors.New("inference: DFlash target model is incompatible")
	}
	if len(tokenIDs) == 0 || len(tokenIDs) != len(positions) || cache == nil || len(cache.Layers) != len(r.weights.Layers) {
		return reference.Value{}, errors.New("inference: DFlash noise block input is incompatible")
	}
	rows := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= target.vocab.Len() {
			return reference.Value{}, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
	}
	activation, err := target.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, err
	}
	for layerIndex, info := range r.weights.Layers {
		activation, _, err = r.runLayerCached(
			ctx, activation, info, layerIndex, positions, nil, &cache.Layers[layerIndex], reference.Value{}, nil,
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
	positions := make([]uint32, draftCount+1)
	ids[0] = last
	for index := range ids {
		positions[index] = cache.Position + uint32(index)
		if index > 0 {
			ids[index] = r.vocab.Mask
		}
	}
	return r.DecodeDFlashNoiseBlock(ctx, target, ids, positions, cache)
}
