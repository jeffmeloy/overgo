package inference

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"

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
	if err := r.validateCache(cache); err != nil {
		return nil, err
	}
	return cache, nil
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
	for index, layer := range cache.Layers {
		if len(layer.States) > maxLayerCacheStates-2 {
			return fmt.Errorf("inference: KV cache layer %d state count exceeds limit", index)
		}
		for name, state := range layer.States {
			if err := validateLayerState(name, state, cache.Tokens); err != nil {
				return fmt.Errorf("inference: KV cache layer %d state %q: %w", index, name, err)
			}
		}
		if r.spec.Architecture == "glm-dsa" {
			state, present := layer.States["indexer_key"]
			if r.spec.LayerHasFullIndexer(uint32(index)) {
				want := tensor.MustShape(uint64(r.spec.IndexerKeyLength), 1, uint64(cache.Tokens))
				if !present || state.Mode != CacheStateToken || !state.Value.Shape.Equal(want) {
					return fmt.Errorf("inference: GLM-DSA cache layer %d indexer shape is invalid", index)
				}
			} else if present {
				return fmt.Errorf("inference: GLM-DSA shared layer %d has indexer state", index)
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
		if r.spec.Architecture == "deepseek2" || r.spec.Architecture == "mistral4" || r.spec.Architecture == "glm-dsa" || r.spec.Architecture == "kimi-linear" {
			kvHeads = uint64(r.spec.HeadCount)
			if index < len(r.weights.Layers) && r.weights.Layers[index].AttentionKB != nil {
				keyWidth = uint64(r.spec.KVLoRARank + r.spec.RopeDimensionCount)
				valueWidth = uint64(r.spec.KVLoRARank)
				kvHeads = 1
			}
		}
		keyShape := tensor.MustShape(keyWidth, kvHeads, uint64(cache.Tokens))
		valueShape := tensor.MustShape(valueWidth, kvHeads, uint64(cache.Tokens))
		if isQwenGDNArchitecture(r.spec.Architecture) &&
			index < len(r.weights.Layers) &&
			r.weights.Layers[index].Recurrent {
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
	if r.spec.Architecture == "t5" {
		return nil, errors.New("inference: T5 relative-bias cache editing is unsupported")
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
	for index, layer := range cache.Layers {
		recurrent := index < len(r.weights.Layers) && r.weights.Layers[index].Recurrent
		states, err := editLayerStates(layer.States, cache.Tokens, start, discard)
		if err != nil {
			return nil, fmt.Errorf("inference: remove cache layer %d named states: %w", index, err)
		}
		if recurrent {
			result.Layers[index] = LayerCache{
				Key:    cloneStateValue(layer.Key),
				Value:  cloneStateValue(layer.Value),
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
	total := uint64(cache.Tokens) + uint64(incoming)
	if total <= uint64(r.spec.ContextLength) || !contextShift {
		return cache, nil
	}
	if uint64(incoming) > uint64(r.spec.ContextLength) {
		return nil, fmt.Errorf(
			"inference: new token count %d exceeds context length %d",
			incoming,
			r.spec.ContextLength,
		)
	}
	discard, err := contextDiscardCount(
		cache.Tokens,
		incoming,
		r.spec.ContextLength,
		keep,
		requestedDiscard,
	)
	if err != nil {
		return nil, err
	}
	return r.RemoveCacheRange(cache, keep, discard)
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

func cloneStateValue(value reference.Value) reference.Value {
	return reference.Value{
		Shape: value.Shape,
		Data:  append([]float32(nil), value.Data...),
	}
}

func cloneCache(cache *KVCache) *KVCache {
	result := &KVCache{
		Layers:   make([]LayerCache, len(cache.Layers)),
		Tokens:   cache.Tokens,
		Position: effectiveCachePosition(cache),
	}
	for index, layer := range cache.Layers {
		result.Layers[index] = LayerCache{
			Key:    cloneStateValue(layer.Key),
			Value:  cloneStateValue(layer.Value),
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
		state.Value = cloneStateValue(state.Value)
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
			state.Value = cloneStateValue(state.Value)
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
		if total > math.MaxUint64-4 {
			return nil, errors.New("inference: KV cache state size overflows")
		}
		total += 4
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
			bytes := uint64(len(record.value.Data)) * 4
			recordSize := uint64(8+len(record.name)) + 44 + bytes
			if total > math.MaxUint64-recordSize {
				return nil, errors.New("inference: KV cache state size overflows")
			}
			total += recordSize
		}
	}
	if total > uint64(maxIntValue()) {
		return nil, errors.New("inference: KV cache state exceeds addressable memory")
	}
	output := make([]byte, int(total))
	copy(output, cacheStateMagic)
	binary.LittleEndian.PutUint32(output[8:], cache.Tokens)
	binary.LittleEndian.PutUint32(output[12:], effectiveCachePosition(cache))
	binary.LittleEndian.PutUint32(output[16:], uint32(len(cache.Layers)))
	offset := cacheStateHeaderSize
	for _, layer := range cache.Layers {
		records := cacheLayerRecords(layer)
		binary.LittleEndian.PutUint32(output[offset:], uint32(len(records)))
		offset += 4
		for _, record := range records {
			binary.LittleEndian.PutUint32(output[offset:], uint32(len(record.name)))
			offset += 4
			copy(output[offset:], record.name)
			offset += len(record.name)
			binary.LittleEndian.PutUint32(output[offset:], uint32(record.mode))
			offset += 4
			writeCacheValue(output, &offset, record.value)
		}
	}
	return output, nil
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

func writeCacheValue(output []byte, offset *int, value reference.Value) {
	binary.LittleEndian.PutUint32(output[*offset:], uint32(value.Shape.Rank))
	*offset += 4
	for _, dimension := range value.Shape.Dims {
		binary.LittleEndian.PutUint64(output[*offset:], dimension)
		*offset += 8
	}
	binary.LittleEndian.PutUint64(output[*offset:], uint64(len(value.Data)))
	*offset += 8
	for _, item := range value.Data {
		binary.LittleEndian.PutUint32(output[*offset:], math.Float32bits(item))
		*offset += 4
	}
}

func unmarshalCache(data []byte) (*KVCache, error) {
	if len(data) < legacyCacheHeaderSize {
		return nil, errors.New("inference: KV cache state is truncated")
	}
	magic := string(data[:8])
	if magic != cacheStateMagic && magic != cacheStateV2Magic && magic != legacyCacheStateMagic {
		return nil, errors.New("inference: KV cache state has invalid magic or version")
	}
	tokens := binary.LittleEndian.Uint32(data[8:])
	position := tokens
	headerSize := legacyCacheHeaderSize
	var layers uint32
	if magic == cacheStateMagic || magic == cacheStateV2Magic {
		if len(data) < cacheStateHeaderSize {
			return nil, errors.New("inference: KV cache state is truncated")
		}
		position = binary.LittleEndian.Uint32(data[12:])
		layers = binary.LittleEndian.Uint32(data[16:])
		headerSize = cacheStateHeaderSize
	} else {
		layers = binary.LittleEndian.Uint32(data[12:])
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
	offset := headerSize
	readValue := func() (reference.Value, error) {
		const header = 44
		if len(data)-offset < header {
			return reference.Value{}, errors.New("inference: KV cache tensor header is truncated")
		}
		rank := binary.LittleEndian.Uint32(data[offset:])
		offset += 4
		if rank == 0 || rank > tensor.MaxDimensions {
			return reference.Value{}, fmt.Errorf("inference: KV cache tensor rank %d is invalid", rank)
		}
		dimensions := make([]uint64, tensor.MaxDimensions)
		for index := range tensor.MaxDimensions {
			dimensions[index] = binary.LittleEndian.Uint64(data[offset:])
			offset += 8
		}
		count := binary.LittleEndian.Uint64(data[offset:])
		offset += 8
		if count > uint64(maxIntValue()) || count > uint64((len(data)-offset)/4) {
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
			values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
			offset += 4
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
		if len(data)-offset < 4 {
			return nil, fmt.Errorf("inference: KV cache layer %d state count is truncated", index)
		}
		stateCount := binary.LittleEndian.Uint32(data[offset:])
		offset += 4
		if stateCount < 2 || stateCount > maxLayerCacheStates {
			return nil, fmt.Errorf("inference: KV cache layer %d state count %d is invalid", index, stateCount)
		}
		seen := make(map[string]struct{}, int(stateCount))
		layer := LayerCache{}
		for stateIndex := uint32(0); stateIndex < stateCount; stateIndex++ {
			if len(data)-offset < 4 {
				return nil, fmt.Errorf("inference: KV cache layer %d state name is truncated", index)
			}
			nameLength := binary.LittleEndian.Uint32(data[offset:])
			offset += 4
			if nameLength == 0 || nameLength > maxCacheStateName || uint64(nameLength) > uint64(len(data)-offset) {
				return nil, fmt.Errorf("inference: KV cache layer %d state name length %d is invalid", index, nameLength)
			}
			name := string(data[offset : offset+int(nameLength)])
			offset += int(nameLength)
			if !validCacheStateName(name) {
				return nil, fmt.Errorf("inference: KV cache layer %d state name %q is invalid", index, name)
			}
			if _, duplicate := seen[name]; duplicate {
				return nil, fmt.Errorf("inference: KV cache layer %d state name %q is duplicated", index, name)
			}
			seen[name] = struct{}{}
			if len(data)-offset < 4 {
				return nil, fmt.Errorf("inference: KV cache layer %d state %q mode is truncated", index, name)
			}
			mode := CacheStateMode(binary.LittleEndian.Uint32(data[offset:]))
			offset += 4
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
	if offset != len(data) {
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

func maxIntValue() int {
	return int(^uint(0) >> 1)
}
