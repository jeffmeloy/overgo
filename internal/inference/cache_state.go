package inference

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

const (
	cacheStateMagic       = "L2GKV002"
	legacyCacheStateMagic = "L2GKV001"
	cacheStateHeaderSize  = 20
	legacyCacheHeaderSize = 16
	maxCacheStateLayers   = 4096
)

// SaveCache serializes attention KV or hybrid recurrent state after validating
// it against the loaded model. Recurrent layers store their convolution and
// delta-net states in LayerCache.Key and LayerCache.Value.
func (r *Runner) SaveCache(cache *KVCache) ([]byte, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateCache(cache); err != nil {
		return nil, err
	}
	return marshalCache(cache)
}

// LoadCache parses untrusted state with bounds checks and validates every
// tensor against the loaded model before returning it.
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
	keyShape := tensor.MustShape(
		uint64(r.spec.KeyLength),
		uint64(r.spec.HeadCountKV),
		uint64(cache.Tokens),
	)
	valueShape := tensor.MustShape(
		uint64(r.spec.ValueLength),
		uint64(r.spec.HeadCountKV),
		uint64(cache.Tokens),
	)
	for index, layer := range cache.Layers {
		if r.spec.Architecture == "lfm2" &&
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
		if r.spec.Architecture == "qwen35" &&
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

// RemoveCacheRange discards one contiguous range of active attention KV
// entries while retaining absolute token positions. Hybrid recurrent layers
// are copied without modification because their fixed-size state summarizes
// the entire history. The input cache and all of its backing slices remain
// independently usable.
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
	for index, layer := range cache.Layers {
		recurrent := index < len(r.weights.Layers) && r.weights.Layers[index].Recurrent
		if recurrent {
			result.Layers[index] = LayerCache{
				Key:   cloneStateValue(layer.Key),
				Value: cloneStateValue(layer.Value),
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
		result.Layers[index] = LayerCache{Key: key, Value: value}
	}
	return result, nil
}

// ShiftCache discards a prefix of active attention KV entries while retaining
// absolute token positions.
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
	// Match the upstream server's safety margin so a keep-all request still
	// leaves room for a shifted suffix and subsequent decode tokens.
	maximum := max(0, int(contextLength)-4)
	keep = min(keep, maximum)
	return uint32(keep)
}

func effectiveCachePosition(cache *KVCache) uint32 {
	if cache == nil {
		return 0
	}
	if cache.Position == 0 {
		// Position was added in cache-state v2. Treat zero on an existing,
		// non-empty in-memory cache as the original append-only representation.
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
			Key:   cloneStateValue(layer.Key),
			Value: cloneStateValue(layer.Value),
		}
	}
	return result
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
	for _, layer := range cache.Layers {
		for _, value := range []reference.Value{layer.Key, layer.Value} {
			if err := validateStateValue(value); err != nil {
				return nil, err
			}
			bytes := uint64(len(value.Data)) * 4
			if total > math.MaxUint64-(44+bytes) {
				return nil, errors.New("inference: KV cache state size overflows")
			}
			total += 44 + bytes
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
		for _, value := range []reference.Value{layer.Key, layer.Value} {
			binary.LittleEndian.PutUint32(output[offset:], uint32(value.Shape.Rank))
			offset += 4
			for _, dimension := range value.Shape.Dims {
				binary.LittleEndian.PutUint64(output[offset:], dimension)
				offset += 8
			}
			binary.LittleEndian.PutUint64(output[offset:], uint64(len(value.Data)))
			offset += 8
			for _, item := range value.Data {
				binary.LittleEndian.PutUint32(output[offset:], math.Float32bits(item))
				offset += 4
			}
		}
	}
	return output, nil
}

func unmarshalCache(data []byte) (*KVCache, error) {
	if len(data) < legacyCacheHeaderSize {
		return nil, errors.New("inference: KV cache state is truncated")
	}
	magic := string(data[:8])
	if magic != cacheStateMagic && magic != legacyCacheStateMagic {
		return nil, errors.New("inference: KV cache state has invalid magic or version")
	}
	tokens := binary.LittleEndian.Uint32(data[8:])
	position := tokens
	headerSize := legacyCacheHeaderSize
	var layers uint32
	if magic == cacheStateMagic {
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
		key, err := readValue()
		if err != nil {
			return nil, fmt.Errorf("inference: KV cache layer %d key: %w", index, err)
		}
		value, err := readValue()
		if err != nil {
			return nil, fmt.Errorf("inference: KV cache layer %d value: %w", index, err)
		}
		result.Layers[index] = LayerCache{Key: key, Value: value}
	}
	if offset != len(data) {
		return nil, errors.New("inference: KV cache state has trailing data")
	}
	return result, nil
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
