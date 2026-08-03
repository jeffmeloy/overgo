package inference

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

	"llamacpp2go/internal/checked"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/statecodec"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

const (
	cacheStateMagic       = "L2GKV003"
	cacheStateV2Magic     = "L2GKV002"
	legacyCacheStateMagic = "L2GKV001"
	cacheStateHeaderSize  = 20
	legacyCacheHeaderSize = 16
	maxCacheStateLayers   = 4096
	maxLayerCacheStates   = 16
	maxCacheStateName     = 64
)

// SaveCache: validated named cache state.
func (r *Runner) SaveCache(cache *KVCache) ([]byte, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateCache(cache); err != nil {
		return nil, err
	}
	return marshalCache(cache)
}

// LoadCache: bounded parse; model-shape validation.
func (r *Runner) LoadCache(data []byte) (*KVCache, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	cache, err := unmarshalCache(data)
	if err != nil {
		return nil, err
	}
	if r.spec.Architecture == "deepseek4" {
		upgradeDeepSeek4CachePositions(cache)
	}
	if err := r.validateCache(cache); err != nil {
		return nil, err
	}
	return cache, nil
}

func upgradeDeepSeek4CachePositions(cache *KVCache) {
	if cache == nil || cache.Tokens == 0 || effectiveCachePosition(cache) < cache.Tokens {
		return
	}
	start := effectiveCachePosition(cache) - cache.Tokens
	data := make([]float32, cache.Tokens)
	for index := range data {
		data[index] = float32(start + uint32(index))
	}
	shape := tensor.MustShape(1, 1, uint64(cache.Tokens))
	for index := range cache.Layers {
		if _, present := cache.Layers[index].States["positions"]; present {
			continue
		}
		if cache.Layers[index].States == nil {
			cache.Layers[index].States = make(map[string]LayerState)
		}
		cache.Layers[index].States["positions"] = LayerState{
			Mode: CacheStateToken, Value: reference.Value{Shape: shape, Data: slices.Clone(data)},
		}
	}
}

func (r *Runner) validateCache(cache *KVCache) error {
	if cache == nil {
		return errors.New("inference: KV cache is nil")
	}
	if cache.Tokens == 0 {
		return errors.New("inference: KV cache token count is zero")
	}
	position := effectiveCachePosition(cache)
	if position < cache.Tokens {
		return fmt.Errorf(
			"inference: KV cache next position %d precedes its %d active tokens",
			position,
			cache.Tokens,
		)
	}
	if r.spec.ContextLength > 0 && cache.Tokens > r.spec.ContextLength {
		return fmt.Errorf(
			"inference: KV cache has %d active tokens, context length is %d",
			cache.Tokens,
			r.spec.ContextLength,
		)
	}
	expectedLayers := int(r.spec.BlockCount)
	if r.spec.Architecture == "t5" {
		expectedLayers = int(r.spec.DecoderBlockCount)
	}
	if expectedLayers == 0 {
		expectedLayers = len(r.weights.Layers)
	}
	if len(cache.Layers) != expectedLayers {
		return fmt.Errorf(
			"inference: KV cache has %d layers, need %d",
			len(cache.Layers),
			expectedLayers,
		)
	}
	var deepSeekPositions []float32
	for index, layer := range cache.Layers {
		if len(layer.States) > maxLayerCacheStates-2 {
			return fmt.Errorf("inference: KV cache layer %d state count exceeds limit", index)
		}
		for name, state := range layer.States {
			if err := validateLayerState(name, state, cache.Tokens); err != nil {
				return fmt.Errorf("inference: KV cache layer %d state %q: %w", index, name, err)
			}
		}
		info := model.LayerWeights{}
		if index < len(r.weights.Layers) {
			info = r.weights.Layers[index]
		}
		schema, err := model.CacheSchema(r.spec, index, info, cache.Tokens)
		if err != nil {
			return fmt.Errorf("inference: KV cache layer %d schema: %w", index, err)
		}
		if err := validateLayerCacheSchema(layer, schema); err != nil {
			return fmt.Errorf("inference: KV cache layer %d: %w", index, err)
		}
		if r.spec.Profile().Attention == model.AttentionDSA {
			state, present := layer.States["indexer_key"]
			if r.spec.LayerHasFullIndexer(uint32(index)) {
				want := tensor.MustShape(uint64(r.spec.IndexerKeyLength), 1, uint64(cache.Tokens))
				if !present || state.Mode != CacheStateToken || !state.Value.Shape.Equal(want) {
					return fmt.Errorf("inference: DSA cache layer %d indexer shape is invalid", index)
				}
			} else if present {
				return fmt.Errorf("inference: DSA shared layer %d has indexer state", index)
			}
		}
		if r.spec.Architecture == "deepseek4" {
			ratio := r.spec.CompressRatios[index]
			expected := map[string]tensor.Shape{
				"positions": tensor.MustShape(1, 1, uint64(cache.Tokens)),
			}
			if ratio != 0 {
				coefficient := uint64(1)
				if ratio == 4 {
					coefficient = 2
				}
				shape := tensor.MustShape(coefficient*uint64(r.spec.KeyLength), 1, uint64(cache.Tokens))
				expected["compressor_kv"] = shape
				expected["compressor_score"] = shape
			}
			if ratio == 4 {
				shape := tensor.MustShape(2*uint64(r.spec.IndexerKeyLength), 1, uint64(cache.Tokens))
				expected["indexer_compressor_kv"] = shape
				expected["indexer_compressor_score"] = shape
			}
			for name, shape := range expected {
				state, present := layer.States[name]
				if !present || state.Mode != CacheStateToken || !state.Value.Shape.Equal(shape) {
					return fmt.Errorf("inference: DeepSeek 4 cache layer %d state %q is invalid", index, name)
				}
			}
			for name := range layer.States {
				if _, present := expected[name]; !present {
					return fmt.Errorf("inference: DeepSeek 4 cache layer %d state %q is unexpected", index, name)
				}
			}
			positions := layer.States["positions"].Value.Data
			for item, value := range positions {
				position := uint32(value)
				if value < 0 || float32(position) != value || position >= effectiveCachePosition(cache) ||
					(item > 0 && position <= uint32(positions[item-1])) {
					return fmt.Errorf("inference: DeepSeek 4 cache layer %d positions are invalid", index)
				}
			}
			if index == 0 {
				deepSeekPositions = slices.Clone(positions)
			} else {
				for item := range positions {
					if positions[item] != deepSeekPositions[item] {
						return fmt.Errorf("inference: DeepSeek 4 cache layer %d positions differ", index)
					}
				}
			}
		}
		if r.spec.Architecture == "falcon-h1" {
			conv, hasConv := layer.States["conv_state"]
			ssm, hasSSM := layer.States["ssm_state"]
			convWidth := uint64(r.spec.SSMInnerSize) + 2*uint64(r.spec.SSMGroupCount)*uint64(r.spec.SSMStateSize)
			convShape := tensor.MustShape(uint64(r.spec.SSMConvKernel-1), convWidth)
			ssmShape := tensor.MustShape(uint64(r.spec.SSMStateSize), uint64(r.spec.SSMInnerSize))
			if !hasConv || conv.Mode != CacheStateFixed || !conv.Value.Shape.Equal(convShape) ||
				!hasSSM || ssm.Mode != CacheStateFixed || !ssm.Value.Shape.Equal(ssmShape) {
				return fmt.Errorf("inference: Falcon-H1 cache layer %d recurrent state is invalid", index)
			}
		}
		if r.spec.Architecture == "t5" {
			crossKey, hasKey := layer.States["cross_key"]
			crossValue, hasValue := layer.States["cross_value"]
			keyShapeOK := hasKey && crossKey.Mode == CacheStateFixed &&
				crossKey.Value.Shape.Rank == 3 &&
				crossKey.Value.Shape.Dims[0] == uint64(r.spec.KeyLength) &&
				crossKey.Value.Shape.Dims[1] == uint64(r.spec.HeadCountKV) &&
				crossKey.Value.Shape.Dims[2] > 0
			valueShapeOK := hasValue && crossValue.Mode == CacheStateFixed &&
				crossValue.Value.Shape.Rank == 3 &&
				crossValue.Value.Shape.Dims[0] == uint64(r.spec.ValueLength) &&
				crossValue.Value.Shape.Dims[1] == uint64(r.spec.HeadCountKV) &&
				crossValue.Value.Shape.Dims[2] > 0
			if !keyShapeOK || !valueShapeOK ||
				crossKey.Value.Shape.Dims[2] != crossValue.Value.Shape.Dims[2] {
				return fmt.Errorf("inference: T5 cache layer %d cross-attention state is invalid", index)
			}
		}
		jambaRecurrent := r.spec.Architecture == "jamba" && index < len(r.weights.Layers) &&
			r.weights.Layers[index].Recurrent
		graniteHybridRecurrent := r.spec.Architecture == "granitehybrid" && index < len(r.weights.Layers) &&
			r.weights.Layers[index].Recurrent
		plamo2Recurrent := r.spec.Architecture == "plamo2" && index < len(r.weights.Layers) &&
			r.weights.Layers[index].Recurrent
		nemotronHRecurrent := (r.spec.Architecture == "nemotron_h" || r.spec.Architecture == "nemotron_h_moe") &&
			index < len(r.weights.Layers) && r.weights.Layers[index].Recurrent
		kimiRecurrent := r.spec.Architecture == "kimi-linear" && index < len(r.weights.Layers) &&
			r.weights.Layers[index].Recurrent
		if r.spec.Architecture == "rwkv6" {
			wantShift := tensor.MustShape(uint64(r.spec.EmbeddingLength), 2)
			wantState := tensor.MustShape(uint64(r.spec.WKVHeadSize), uint64(r.spec.WKVHeadSize), uint64(r.spec.HeadCount), 1)
			if !layer.Key.Shape.Equal(wantShift) || !layer.Value.Shape.Equal(wantState) {
				return fmt.Errorf("inference: RWKV6 cache layer %d shape is invalid", index)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: RWKV6 cache layer %d shift: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: RWKV6 cache layer %d state: %w", index, err)
			}
			continue
		}
		if r.spec.Architecture == "rwkv6qwen2" {
			wantShift := tensor.MustShape(uint64(r.spec.EmbeddingLength))
			wantState := tensor.MustShape(uint64(r.spec.WKVHeadSize), uint64(r.spec.WKVHeadSize), uint64(r.spec.HeadCount), 1)
			if !layer.Key.Shape.Equal(wantShift) || !layer.Value.Shape.Equal(wantState) {
				return fmt.Errorf("inference: RWKV6-Qwen2 cache layer %d shape is invalid", index)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: RWKV6-Qwen2 cache layer %d shift: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: RWKV6-Qwen2 cache layer %d state: %w", index, err)
			}
			continue
		}
		if r.spec.Architecture == "rwkv7" || r.spec.Architecture == "arwkv7" {
			wantShift := tensor.MustShape(uint64(r.spec.EmbeddingLength), uint64(r.spec.TokenShiftCount))
			wantState := tensor.MustShape(uint64(r.spec.WKVHeadSize), uint64(r.spec.WKVHeadSize), uint64(r.spec.HeadCount), 1)
			if !layer.Key.Shape.Equal(wantShift) || !layer.Value.Shape.Equal(wantState) {
				return fmt.Errorf("inference: RWKV7 cache layer %d shape is invalid", index)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: RWKV7 cache layer %d shift: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: RWKV7 cache layer %d state: %w", index, err)
			}
			continue
		}
		if kimiRecurrent {
			convShape := tensor.MustShape(uint64(r.spec.SSMConvKernel-1), 3*uint64(r.spec.SSMInnerSize))
			ssmShape := tensor.MustShape(uint64(r.spec.KDAHeadDim), uint64(r.spec.KDAHeadDim), uint64(r.spec.HeadCount), 1)
			if !layer.Key.Shape.Equal(convShape) || !layer.Value.Shape.Equal(ssmShape) {
				return fmt.Errorf("inference: Kimi Linear recurrent cache layer %d shape is invalid", index)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: Kimi Linear recurrent cache layer %d convolution: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: Kimi Linear recurrent cache layer %d state: %w", index, err)
			}
			continue
		}
		if r.spec.Architecture == "mamba" || r.spec.Architecture == "mamba2" || jambaRecurrent || graniteHybridRecurrent || plamo2Recurrent || nemotronHRecurrent {
			convWidth := uint64(r.spec.SSMInnerSize)
			if r.spec.Architecture == "mamba2" || graniteHybridRecurrent || nemotronHRecurrent {
				convWidth += 2 * uint64(r.spec.SSMGroupCount) * uint64(r.spec.SSMStateSize)
			}
			convShape := tensor.MustShape(
				uint64(r.spec.SSMConvKernel-1), convWidth,
			)
			ssmShape := tensor.MustShape(
				uint64(r.spec.SSMStateSize), uint64(r.spec.SSMInnerSize),
			)
			if !layer.Key.Shape.Equal(convShape) || !layer.Value.Shape.Equal(ssmShape) {
				return fmt.Errorf("inference: Mamba recurrent cache layer %d shape is invalid", index)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: Mamba recurrent cache layer %d convolution: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: Mamba recurrent cache layer %d SSM: %w", index, err)
			}
			continue
		}
		if (r.spec.Architecture == "nemotron_h" || r.spec.Architecture == "nemotron_h_moe") &&
			r.spec.LayerFeedForwardLength(uint32(index)) > 0 {
			sentinelShape := tensor.MustShape(1, 1, uint64(cache.Tokens))
			if !layer.Key.Shape.Equal(sentinelShape) || !layer.Value.Shape.Equal(sentinelShape) {
				return fmt.Errorf("inference: Nemotron-H sentinel cache layer %d shape is invalid", index)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: Nemotron-H sentinel cache layer %d key: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: Nemotron-H sentinel cache layer %d value: %w", index, err)
			}
			continue
		}
		if (r.spec.Architecture == "lfm2" || r.spec.Architecture == "lfm2moe") &&
			index < len(r.weights.Layers) && r.weights.Layers[index].Recurrent {
			convShape := tensor.MustShape(
				uint64(r.spec.ShortConvCacheLength-1), uint64(r.spec.EmbeddingLength),
			)
			if !layer.Key.Shape.Equal(convShape) || !layer.Value.Shape.Equal(tensor.MustShape(1)) {
				return fmt.Errorf("inference: LFM2 recurrent cache layer %d shape is invalid", index)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: LFM2 recurrent cache layer %d convolution: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: LFM2 recurrent cache layer %d reserved state: %w", index, err)
			}
			continue
		}
		if r.spec.Architecture == "deci" && r.spec.LayerKVHeadCount(uint32(index)) == 0 {
			sentinelShape := tensor.MustShape(1, 1, uint64(cache.Tokens))
			if !layer.Key.Shape.Equal(sentinelShape) || !layer.Value.Shape.Equal(sentinelShape) {
				return fmt.Errorf("inference: Deci sentinel cache layer %d shape is invalid", index)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: Deci sentinel cache layer %d key: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: Deci sentinel cache layer %d value: %w", index, err)
			}
			continue
		}
		keyWidth := uint64(r.spec.LayerKeyLength(uint32(index)))
		valueWidth := uint64(r.spec.LayerValueLength(uint32(index)))
		kvHeads := uint64(r.spec.LayerKVHeadCount(uint32(index)))
		if r.spec.Architecture == "deepseek2" || r.spec.Architecture == "deepseek32" || r.spec.Architecture == "mistral4" || r.spec.Architecture == "glm-dsa" || r.spec.Architecture == "kimi-linear" {
			kvHeads = uint64(r.spec.HeadCount)
			if index < len(r.weights.Layers) && r.weights.Layers[index].AttentionKB != nil {
				keyWidth = uint64(r.spec.KVLoRARank + r.spec.RopeDimensionCount)
				valueWidth = uint64(r.spec.KVLoRARank)
				kvHeads = 1
			}
		}
		keyShape := tensor.MustShape(keyWidth, kvHeads, uint64(cache.Tokens))
		valueShape := tensor.MustShape(valueWidth, kvHeads, uint64(cache.Tokens))
		qwenRecurrent := r.spec.IsRecurrentLayer(uint32(index)) ||
			index < len(r.weights.Layers) && r.weights.Layers[index].Recurrent
		if r.spec.Profile().Attention == model.AttentionQwenGDN && qwenRecurrent {
			convChannels := uint64(r.spec.SSMInnerSize) +
				2*uint64(r.spec.SSMStateSize)*uint64(r.spec.SSMGroupCount)
			convShape := tensor.MustShape(
				uint64(r.spec.SSMConvKernel-1),
				convChannels,
			)
			ssmShape := tensor.MustShape(
				uint64(r.spec.SSMStateSize),
				uint64(r.spec.SSMStateSize),
				uint64(r.spec.SSMTimeStepRank),
				1,
			)
			if !layer.Key.Shape.Equal(convShape) {
				return fmt.Errorf(
					"inference: recurrent cache layer %d convolution shape %v, need %v",
					index,
					layer.Key.Shape.Slice(),
					convShape.Slice(),
				)
			}
			if !layer.Value.Shape.Equal(ssmShape) {
				return fmt.Errorf(
					"inference: recurrent cache layer %d state shape %v, need %v",
					index,
					layer.Value.Shape.Slice(),
					ssmShape.Slice(),
				)
			}
			if err := validateStateValue(layer.Key); err != nil {
				return fmt.Errorf("inference: recurrent cache layer %d convolution: %w", index, err)
			}
			if err := validateStateValue(layer.Value); err != nil {
				return fmt.Errorf("inference: recurrent cache layer %d state: %w", index, err)
			}
			continue
		}
		if !layer.Key.Shape.Equal(keyShape) {
			return fmt.Errorf(
				"inference: KV cache layer %d key shape %v, need %v",
				index,
				layer.Key.Shape.Slice(),
				keyShape.Slice(),
			)
		}
		if !layer.Value.Shape.Equal(valueShape) {
			return fmt.Errorf(
				"inference: KV cache layer %d value shape %v, need %v",
				index,
				layer.Value.Shape.Slice(),
				valueShape.Slice(),
			)
		}
		if err := validateStateValue(layer.Key); err != nil {
			return fmt.Errorf("inference: KV cache layer %d key: %w", index, err)
		}
		if err := validateStateValue(layer.Value); err != nil {
			return fmt.Errorf("inference: KV cache layer %d value: %w", index, err)
		}
	}
	return nil
}

func validateLayerCacheSchema(layer LayerCache, schema model.LayerCacheSchema) error {
	for name, expected := range schema.States {
		state, present := layer.States[name]
		if !present {
			return fmt.Errorf("required state %q is missing", name)
		}
		mode := CacheStateFixed
		if expected.Extent == model.CacheExtentToken {
			mode = CacheStateToken
		}
		if state.Mode != mode || !cacheShapeMatches(state.Value.Shape, expected) {
			return fmt.Errorf("state %q shape or extent is invalid", name)
		}
	}
	if schema.StrictStates {
		for name := range layer.States {
			if _, present := schema.States[name]; !present {
				return fmt.Errorf("state %q is unexpected", name)
			}
		}
	}
	for index, value := range []reference.Value{layer.Key, layer.Value} {
		expected := schema.Primary[index]
		if !cacheShapeMatches(value.Shape, expected) {
			return fmt.Errorf(
				"%s state %d shape %v, need %v",
				schema.Label, index, value.Shape.Slice(), expected.Shape.Slice(),
			)
		}
		if err := validateStateValue(value); err != nil {
			return fmt.Errorf("%s state %d: %w", schema.Label, index, err)
		}
	}
	return nil
}

func cacheShapeMatches(shape tensor.Shape, schema model.CacheValueSchema) bool {
	if !schema.VariableLast {
		return shape.Equal(schema.Shape)
	}
	if shape.Rank != schema.Shape.Rank || shape.Rank == 0 {
		return false
	}
	last := int(shape.Rank) - 1
	for index := 0; index < last; index++ {
		if shape.Dims[index] != schema.Shape.Dims[index] {
			return false
		}
	}
	return shape.Dims[last] > 0
}

func (r *Runner) validateT5Cache(cache *KVCache, encoderTokens uint64) error {
	if err := r.validateCache(cache); err != nil {
		return err
	}
	for index, layer := range cache.Layers {
		if layer.States["cross_key"].Value.Shape.Dims[2] != encoderTokens {
			return fmt.Errorf(
				"inference: T5 cache layer %d encoder length %d, need %d",
				index, layer.States["cross_key"].Value.Shape.Dims[2], encoderTokens,
			)
		}
	}
	return nil
}

// RemoveCacheRange: active-range delete; absolute position retained.
func (r *Runner) RemoveCacheRange(
	cache *KVCache,
	start, discard uint32,
) (*KVCache, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateCache(cache); err != nil {
		return nil, err
	}
	if discard == 0 {
		return cloneCache(cache), nil
	}
	end := uint64(start) + uint64(discard)
	if start >= cache.Tokens ||
		end > uint64(cache.Tokens) ||
		discard >= cache.Tokens {
		return nil, fmt.Errorf(
			"inference: cache range [%d,%d) is invalid for a %d-token cache; at least one token must remain",
			start,
			end,
			cache.Tokens,
		)
	}
	remaining := cache.Tokens - discard
	result := &KVCache{
		Layers:   make([]LayerCache, len(cache.Layers)),
		Tokens:   remaining,
		Position: effectiveCachePosition(cache),
	}
	if r.spec.Architecture == "t5" {
		// T5 relative positions: translation-invariant; compact rows.
		result.Position = remaining
	}
	for index, layer := range cache.Layers {
		recurrent := index < len(r.weights.Layers) && r.weights.Layers[index].Recurrent
		states, err := editLayerStates(layer.States, cache.Tokens, start, discard)
		if err != nil {
			return nil, fmt.Errorf("inference: remove cache layer %d named states: %w", index, err)
		}
		if recurrent {
			result.Layers[index] = LayerCache{
				Key:    layer.Key.Clone(),
				Value:  layer.Value.Clone(),
				States: states,
			}
			continue
		}
		key, err := removeAttentionRange(
			layer.Key,
			cache.Tokens,
			start,
			discard,
		)
		if err != nil {
			return nil, fmt.Errorf("inference: remove cache layer %d key range: %w", index, err)
		}
		value, err := removeAttentionRange(
			layer.Value,
			cache.Tokens,
			start,
			discard,
		)
		if err != nil {
			return nil, fmt.Errorf("inference: remove cache layer %d value range: %w", index, err)
		}
		result.Layers[index] = LayerCache{Key: key, Value: value, States: states}
	}
	return result, nil
}

// ShiftCache: prefix delete; absolute position retained.
func (r *Runner) ShiftCache(cache *KVCache, discard uint32) (*KVCache, error) {
	return r.RemoveCacheRange(cache, 0, discard)
}

func (r *Runner) cacheForAppend(
	cache *KVCache,
	incoming int,
	contextShift bool,
) (*KVCache, error) {
	return r.cacheForAppendKeeping(cache, incoming, contextShift, 0, -1)
}

func (r *Runner) cacheForAppendKeeping(
	cache *KVCache,
	incoming int,
	contextShift bool,
	keep uint32,
	requestedDiscard int,
) (*KVCache, error) {
	if cache == nil || incoming <= 0 {
		return cache, nil
	}
	discard, needed, err := planContextShift(
		cache.Tokens, incoming, r.spec.ContextLength, keep, requestedDiscard, contextShift,
	)
	if err != nil {
		return nil, err
	}
	if !needed {
		return cache, nil
	}
	return r.RemoveCacheRange(cache, keep, discard)
}

func planContextShift(
	tokens uint32,
	incoming int,
	contextLength, keep uint32,
	requested int,
	enabled bool,
) (discard uint32, needed bool, err error) {
	if incoming <= 0 || uint64(tokens)+uint64(incoming) <= uint64(contextLength) || !enabled {
		return 0, false, nil
	}
	if uint64(incoming) > uint64(contextLength) {
		return 0, false, fmt.Errorf(
			"inference: new token count %d exceeds context length %d", incoming, contextLength,
		)
	}
	discard, err = contextDiscardCount(tokens, incoming, contextLength, keep, requested)
	return discard, err == nil, err
}

func contextDiscardCount(
	tokens uint32,
	incoming int,
	contextLength, keep uint32,
	requested int,
) (uint32, error) {
	if keep >= tokens {
		return 0, fmt.Errorf(
			"inference: cannot preserve %d initial tokens in a %d-token cache",
			keep,
			tokens,
		)
	}
	minimum := uint64(tokens) + uint64(incoming) - uint64(contextLength)
	maximum := uint64(tokens - keep - 1)
	if minimum > maximum {
		return 0, fmt.Errorf(
			"inference: cannot preserve %d initial tokens while fitting %d new tokens in a %d-token context",
			keep,
			incoming,
			contextLength,
		)
	}
	discard := minimum
	switch {
	case requested > 0:
		discard = max(discard, uint64(requested))
	case requested == 0:
		discard = max(discard, uint64(tokens-keep)/2)
	}
	discard = min(discard, maximum)
	return uint32(discard), nil
}

func effectiveKeepTokens(requested, promptTokens int, contextLength uint32) uint32 {
	if requested == 0 || promptTokens <= 0 || contextLength == 0 {
		return 0
	}
	keep := requested
	if keep < 0 || keep > promptTokens {
		keep = promptTokens
	}
	// Upstream-compatible four-token safety margin.
	maximum := max(0, int(contextLength)-4)
	keep = min(keep, maximum)
	return uint32(keep)
}

func effectiveCachePosition(cache *KVCache) uint32 {
	if cache == nil {
		return 0
	}
	if cache.Position == 0 {
		// v1/in-memory zero: append-only position.
		return cache.Tokens
	}
	return cache.Position
}

func cloneCache(cache *KVCache) *KVCache {
	result := &KVCache{
		Layers:   make([]LayerCache, len(cache.Layers)),
		Tokens:   cache.Tokens,
		Position: effectiveCachePosition(cache),
	}
	for index, layer := range cache.Layers {
		result.Layers[index] = LayerCache{
			Key:    layer.Key.Clone(),
			Value:  layer.Value.Clone(),
			States: cloneLayerStates(layer.States),
		}
	}
	return result
}

func cloneLayerStates(states map[string]LayerState) map[string]LayerState {
	if states == nil {
		return nil
	}
	result := make(map[string]LayerState, len(states))
	for name, state := range states {
		state.Value = state.Value.Clone()
		result[name] = state
	}
	return result
}

func editLayerStates(
	states map[string]LayerState,
	tokens, start, discard uint32,
) (map[string]LayerState, error) {
	if states == nil {
		return nil, nil
	}
	result := make(map[string]LayerState, len(states))
	for name, state := range states {
		switch state.Mode {
		case CacheStateFixed:
			state.Value = state.Value.Clone()
		case CacheStateToken:
			value, err := removeAttentionRange(state.Value, tokens, start, discard)
			if err != nil {
				return nil, fmt.Errorf("state %q: %w", name, err)
			}
			state.Value = value
		default:
			return nil, fmt.Errorf("state %q has invalid mode %d", name, state.Mode)
		}
		result[name] = state
	}
	return result, nil
}

func removeAttentionRange(
	value reference.Value,
	tokens, start, discard uint32,
) (reference.Value, error) {
	if value.Shape.Rank != 3 || value.Shape.Dims[2] != uint64(tokens) {
		return reference.Value{}, fmt.Errorf(
			"tensor shape %v does not contain %d cache tokens",
			value.Shape.Slice(),
			tokens,
		)
	}
	stride := value.Shape.Dims[0] * value.Shape.Dims[1]
	firstEnd := uint64(start) * stride
	secondStart := uint64(start+discard) * stride
	if firstEnd > uint64(len(value.Data)) ||
		secondStart > uint64(len(value.Data)) {
		return reference.Value{}, errors.New("tensor data is shorter than its cache shape")
	}
	remaining := tokens - discard
	count := uint64(remaining) * stride
	shape := value.Shape
	shape.Dims[2] = uint64(remaining)
	data := make([]float32, 0, int(count))
	data = append(data, value.Data[:int(firstEnd)]...)
	data = append(data, value.Data[int(secondStart):]...)
	return reference.Value{Shape: shape, Data: data}, nil
}

func marshalCache(cache *KVCache) ([]byte, error) {
	if cache == nil {
		return nil, errors.New("inference: KV cache is nil")
	}
	if len(cache.Layers) > maxCacheStateLayers {
		return nil, errors.New("inference: KV cache layer count exceeds state limit")
	}
	total := uint64(cacheStateHeaderSize)
	for index, layer := range cache.Layers {
		if len(layer.States) > maxLayerCacheStates-2 {
			return nil, fmt.Errorf("inference: KV cache layer %d state count exceeds limit", index)
		}
		var ok bool
		total, ok = checked.Add64(total, 4)
		if !ok {
			return nil, errors.New("inference: KV cache state size overflows")
		}
		records := cacheLayerRecords(layer)
		for _, record := range records {
			if record.mode != 0 {
				if err := validateLayerState(record.name, LayerState{
					Mode: record.mode, Value: record.value,
				}, cache.Tokens); err != nil {
					return nil, fmt.Errorf("inference: KV cache layer %d state %q: %w", index, record.name, err)
				}
			} else if err := validateStateValue(record.value); err != nil {
				return nil, fmt.Errorf("inference: KV cache layer %d state %q: %w", index, record.name, err)
			}
			bytes, ok := checked.Bytes(uint64(len(record.value.Data)), 4)
			recordSize, okSize := checked.Add64(8, uint64(len(record.name)), 44, bytes)
			if !ok || !okSize {
				return nil, errors.New("inference: KV cache state size overflows")
			}
			total, ok = checked.Add64(total, recordSize)
			if !ok {
				return nil, errors.New("inference: KV cache state size overflows")
			}
		}
	}
	if total > uint64(math.MaxInt) {
		return nil, errors.New("inference: KV cache state exceeds addressable memory")
	}
	encoder := statecodec.NewEncoderCapacity(uint64(math.MaxInt), total)
	encoder.Raw([]byte(cacheStateMagic))
	encoder.U32(cache.Tokens)
	encoder.U32(effectiveCachePosition(cache))
	encoder.U32(uint32(len(cache.Layers)))
	for _, layer := range cache.Layers {
		records := cacheLayerRecords(layer)
		encoder.U32(uint32(len(records)))
		for _, record := range records {
			encoder.String32(record.name)
			encoder.U32(uint32(record.mode))
			writeCacheValue(encoder, record.value)
		}
	}
	return encoder.Data()
}

type cacheLayerRecord struct {
	name  string
	mode  CacheStateMode
	value reference.Value
}

func cacheLayerRecords(layer LayerCache) []cacheLayerRecord {
	records := make([]cacheLayerRecord, 0, 2+len(layer.States))
	records = append(records,
		cacheLayerRecord{name: "key", value: layer.Key},
		cacheLayerRecord{name: "value", value: layer.Value},
	)
	names := make([]string, 0, len(layer.States))
	for name := range layer.States {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		state := layer.States[name]
		records = append(records, cacheLayerRecord{name: name, mode: state.Mode, value: state.Value})
	}
	return records
}

func writeCacheValue(encoder *statecodec.Encoder, value reference.Value) {
	encoder.U32(uint32(value.Shape.Rank))
	for _, dimension := range value.Shape.Dims {
		encoder.U64(dimension)
	}
	encoder.U64(uint64(len(value.Data)))
	for _, item := range value.Data {
		encoder.F32(item)
	}
}

func unmarshalCache(data []byte) (*KVCache, error) {
	decoder := statecodec.NewDecoder(data, uint64(math.MaxInt))
	magic := string(decoder.Raw(8))
	if magic != cacheStateMagic && magic != cacheStateV2Magic && magic != legacyCacheStateMagic {
		return nil, errors.New("inference: KV cache state has invalid magic or version")
	}
	tokens := decoder.U32()
	position := tokens
	var layers uint32
	if magic == cacheStateMagic || magic == cacheStateV2Magic {
		position = decoder.U32()
		layers = decoder.U32()
	} else {
		layers = decoder.U32()
	}
	if decoder.Err() != nil {
		return nil, errors.New("inference: KV cache state is truncated")
	}
	if layers > maxCacheStateLayers {
		return nil, errors.New("inference: KV cache state layer count exceeds limit")
	}
	result := &KVCache{
		Layers:   make([]LayerCache, int(layers)),
		Tokens:   tokens,
		Position: position,
	}
	if result.Position == 0 && result.Tokens != 0 {
		result.Position = result.Tokens
	}
	readValue := func() (reference.Value, error) {
		rank := decoder.U32()
		if rank == 0 || rank > tensor.MaxDimensions {
			return reference.Value{}, fmt.Errorf("inference: KV cache tensor rank %d is invalid", rank)
		}
		dimensions := make([]uint64, tensor.MaxDimensions)
		for index := range tensor.MaxDimensions {
			dimensions[index] = decoder.U64()
		}
		count := decoder.U64()
		bytes, ok := checked.Bytes(count, 4)
		if decoder.Err() != nil || !ok || count > uint64(math.MaxInt) || bytes > decoder.Remaining() {
			return reference.Value{}, errors.New("inference: KV cache tensor data is truncated or too large")
		}
		shape, err := tensor.NewShape(dimensions[:rank]...)
		if err != nil {
			return reference.Value{}, fmt.Errorf("inference: KV cache tensor shape: %w", err)
		}
		elements, err := shape.Elements()
		if err != nil || elements != count {
			return reference.Value{}, errors.New("inference: KV cache tensor element count differs from shape")
		}
		values := make([]float32, int(count))
		for index := range values {
			values[index] = decoder.F32()
		}
		return reference.Value{Shape: shape, Data: values}, nil
	}
	for index := range result.Layers {
		if magic != cacheStateMagic {
			key, err := readValue()
			if err != nil {
				return nil, fmt.Errorf("inference: KV cache layer %d key: %w", index, err)
			}
			value, err := readValue()
			if err != nil {
				return nil, fmt.Errorf("inference: KV cache layer %d value: %w", index, err)
			}
			result.Layers[index] = LayerCache{Key: key, Value: value}
			continue
		}
		stateCount := decoder.U32()
		if decoder.Err() != nil {
			return nil, fmt.Errorf("inference: KV cache layer %d state count is truncated", index)
		}
		if stateCount < 2 || stateCount > maxLayerCacheStates {
			return nil, fmt.Errorf("inference: KV cache layer %d state count %d is invalid", index, stateCount)
		}
		seen := make(map[string]struct{}, int(stateCount))
		layer := LayerCache{}
		for stateIndex := uint32(0); stateIndex < stateCount; stateIndex++ {
			name := decoder.String32(maxCacheStateName)
			if decoder.Err() != nil || name == "" {
				return nil, fmt.Errorf("inference: KV cache layer %d state name is invalid", index)
			}
			if !validCacheStateName(name) {
				return nil, fmt.Errorf("inference: KV cache layer %d state name %q is invalid", index, name)
			}
			if _, duplicate := seen[name]; duplicate {
				return nil, fmt.Errorf("inference: KV cache layer %d state name %q is duplicated", index, name)
			}
			seen[name] = struct{}{}
			mode := CacheStateMode(decoder.U32())
			if decoder.Err() != nil {
				return nil, fmt.Errorf("inference: KV cache layer %d state %q mode is truncated", index, name)
			}
			value, err := readValue()
			if err != nil {
				return nil, fmt.Errorf("inference: KV cache layer %d state %q: %w", index, name, err)
			}
			switch name {
			case "key":
				if mode != 0 {
					return nil, fmt.Errorf("inference: KV cache layer %d key mode %d is invalid", index, mode)
				}
				layer.Key = value
			case "value":
				if mode != 0 {
					return nil, fmt.Errorf("inference: KV cache layer %d value mode %d is invalid", index, mode)
				}
				layer.Value = value
			default:
				state := LayerState{Mode: mode, Value: value}
				if err := validateLayerState(name, state, tokens); err != nil {
					return nil, fmt.Errorf("inference: KV cache layer %d state %q: %w", index, name, err)
				}
				if layer.States == nil {
					layer.States = make(map[string]LayerState)
				}
				layer.States[name] = state
			}
		}
		if _, found := seen["key"]; !found {
			return nil, fmt.Errorf("inference: KV cache layer %d key state is missing", index)
		}
		if _, found := seen["value"]; !found {
			return nil, fmt.Errorf("inference: KV cache layer %d value state is missing", index)
		}
		result.Layers[index] = layer
	}
	if err := decoder.Done(); err != nil {
		return nil, errors.New("inference: KV cache state has trailing data")
	}
	return result, nil
}

func validateLayerState(name string, state LayerState, tokens uint32) error {
	if !validCacheStateName(name) || name == "key" || name == "value" {
		return errors.New("invalid name")
	}
	if state.Mode != CacheStateFixed && state.Mode != CacheStateToken {
		return fmt.Errorf("invalid mode %d", state.Mode)
	}
	if err := validateStateValue(state.Value); err != nil {
		return err
	}
	if state.Mode == CacheStateToken &&
		(state.Value.Shape.Rank != 3 || state.Value.Shape.Dims[2] != uint64(tokens)) {
		return fmt.Errorf("token-aligned shape %v does not contain %d tokens", state.Value.Shape.Slice(), tokens)
	}
	return nil
}

func validCacheStateName(name string) bool {
	if len(name) == 0 || len(name) > maxCacheStateName {
		return false
	}
	for index := range len(name) {
		character := name[index]
		if (character >= 'a' && character <= 'z') ||
			(character >= '0' && character <= '9') ||
			character == '_' || character == '-' || character == '.' {
			continue
		}
		return false
	}
	return true
}

func validateStateValue(value reference.Value) error {
	elements, err := value.Shape.Elements()
	if err != nil {
		return err
	}
	if elements != uint64(len(value.Data)) {
		return fmt.Errorf(
			"tensor has %d values, shape requires %d",
			len(value.Data),
			elements,
		)
	}
	return nil
}
