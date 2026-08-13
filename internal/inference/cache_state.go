package inference

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/statecodec"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const (
	cacheStateMagic        = "L2GKV003"
	cacheStateV2Magic      = "L2GKV002"
	legacyCacheStateMagic  = "L2GKV001"
	cacheStateHeaderSize   = 20
	legacyCacheHeaderSize  = 16
	maxCacheStateLayers    = 4096
	maxLayerCacheStates    = 16
	maxCacheStateName      = 64
	cacheLayerCountBytes   = 4
	cacheRecordPrefixBytes = 8
	cacheRecordValueBytes  = 44
	cacheScalarBytes       = 4
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
	if r.hasCachePolicy(model.CacheCompressedAttention) {
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
		if _, present := cache.Layers[index].States[model.CacheStatePositions]; present {
			continue
		}
		if cache.Layers[index].States == nil {
			cache.Layers[index].States = make(LayerStates)
		}
		cache.Layers[index].States[model.CacheStatePositions] = LayerState{
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
	expectedLayers := int(r.program.Model.CacheLayerCount())
	if expectedLayers == 0 {
		return errors.New("inference: model program has no cache layers")
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
		plan, schema, err := r.cacheSchema(index, cache.Tokens)
		if err != nil {
			return fmt.Errorf("inference: KV cache layer %d schema: %w", index, err)
		}
		if err := validateLayerCacheSchema(layer, schema); err != nil {
			return fmt.Errorf("inference: KV cache layer %d: %w", index, err)
		}
		if plan.Attention == model.AttentionSparseLatent {
			state, present := layer.States[model.CacheStateIndexerKey]
			if r.spec.LayerHasFullIndexer(uint32(index)) {
				want := tensor.MustShape(uint64(r.spec.IndexerKeyLength), 1, uint64(cache.Tokens))
				if !present || !state.Mode.TokenAligned() || !state.Value.Shape.Equal(want) {
					return fmt.Errorf("inference: DSA cache layer %d indexer shape is invalid", index)
				}
			} else if present {
				return fmt.Errorf("inference: DSA shared layer %d has indexer state", index)
			}
		}
		if plan.Cache == model.CacheCompressedAttention {
			positions := layer.States[model.CacheStatePositions].Value.Data
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
		if plan.Cache == model.CacheCrossAttention {
			crossKey := layer.States[model.CacheStateCrossKey].Value.Shape.Dims[2]
			crossValue := layer.States[model.CacheStateCrossValue].Value.Shape.Dims[2]
			if crossKey != crossValue {
				return fmt.Errorf(
					"inference: encoder-decoder cache layer %d cross-attention lengths differ",
					index,
				)
			}
		}
	}
	return nil
}

func (r *Runner) hasCachePolicy(policy model.CachePolicy) bool {
	return r.program.Model.HasCache(policy)
}

func (r *Runner) cacheSchema(
	layer int,
	tokens uint32,
) (model.LayerPlan, model.LayerCacheSchema, error) {
	plan := r.layerPlan(layer)
	schema, err := r.program.Model.CacheSchema(layer, tokens)
	return plan, schema, err
}

func validateLayerCacheSchema(layer LayerCache, schema model.LayerCacheSchema) error {
	var validationErr error
	schema.RangeStates(func(name model.CacheStateName, expected model.CacheState[model.CacheValueSchema]) {
		if validationErr != nil {
			return
		}
		state, present := layer.States[name]
		if !present {
			validationErr = fmt.Errorf("required state %q is missing", name)
			return
		}
		if state.Mode != expected.Mode || !cacheShapeMatches(state.Value.Shape, expected.Value) {
			validationErr = fmt.Errorf("state %q shape or mode is invalid", name)
		}
	})
	if validationErr != nil {
		return validationErr
	}
	if schema.StrictStates {
		for name := range layer.States {
			if !schema.HasState(name) {
				return fmt.Errorf("state %q is unexpected", name)
			}
		}
	}
	for _, item := range []struct {
		name     string
		value    reference.Value
		expected model.CacheValueSchema
	}{
		{"key", layer.Key, schema.Primary.Key.Value},
		{"value", layer.Value, schema.Primary.Value.Value},
	} {
		if !cacheShapeMatches(item.value.Shape, item.expected) {
			return fmt.Errorf(
				"%s %s shape %v, need %v",
				schema.Label, item.name, item.value.Shape.Slice(), item.expected.Shape.Slice(),
			)
		}
		if err := validateStateValue(item.value); err != nil {
			return fmt.Errorf("%s %s: %w", schema.Label, item.name, err)
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

func (r *Runner) validateEncoderDecoderCache(cache *KVCache, encoderTokens uint64) error {
	if err := r.validateCache(cache); err != nil {
		return err
	}
	for index, layer := range cache.Layers {
		if layer.States[model.CacheStateCrossKey].Value.Shape.Dims[2] != encoderTokens {
			return fmt.Errorf(
				"inference: encoder-decoder cache layer %d encoder length %d, need %d",
				index, layer.States[model.CacheStateCrossKey].Value.Shape.Dims[2], encoderTokens,
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
	if r.profile().Family == model.ArchitectureFamilyEncoderDecoder {
		// T5 relative positions: translation-invariant; compact rows.
		result.Position = remaining
	}
	for index, layer := range cache.Layers {
		_, schema, err := r.cacheSchema(index, cache.Tokens)
		if err != nil {
			return nil, fmt.Errorf("inference: remove cache layer %d schema: %w", index, err)
		}
		states, err := editLayerStates(layer.States, cache.Tokens, start, discard)
		if err != nil {
			return nil, fmt.Errorf("inference: remove cache layer %d named states: %w", index, err)
		}
		key, err := editPrimaryCacheValue(
			layer.Key, schema.Primary.Key.Mode, cache.Tokens, start, discard,
		)
		if err != nil {
			return nil, fmt.Errorf("inference: remove cache layer %d key range: %w", index, err)
		}
		value, err := editPrimaryCacheValue(
			layer.Value, schema.Primary.Value.Mode, cache.Tokens, start, discard,
		)
		if err != nil {
			return nil, fmt.Errorf("inference: remove cache layer %d value range: %w", index, err)
		}
		result.Layers[index] = LayerCache{Key: key, Value: value, States: states}
	}
	return result, nil
}

func editPrimaryCacheValue(
	value reference.Value,
	mode model.CacheStateMode,
	tokens, start, discard uint32,
) (reference.Value, error) {
	if mode == model.CacheStateFixed {
		return value.Clone(), nil
	}
	return removeAttentionRange(value, tokens, start, discard)
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
			Key:   layer.Key.Clone(),
			Value: layer.Value.Clone(),
			States: layer.States.CloneValues(func(value reference.Value) reference.Value {
				return value.Clone()
			}),
		}
	}
	return result
}

func editLayerStates(
	states LayerStates,
	tokens, start, discard uint32,
) (LayerStates, error) {
	if states == nil {
		return nil, nil
	}
	result := make(LayerStates, len(states))
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
		total, ok = checked.Add64(total, cacheLayerCountBytes)
		if !ok {
			return nil, errors.New("inference: KV cache state size overflows")
		}
		records := cacheLayerRecords(layer)
		for _, record := range records {
			if record.mode != 0 {
				if err := validateLayerState(model.CacheStateName(record.name), LayerState{
					Mode: record.mode, Value: record.value,
				}, cache.Tokens); err != nil {
					return nil, fmt.Errorf("inference: KV cache layer %d state %q: %w", index, record.name, err)
				}
			} else if err := validateStateValue(record.value); err != nil {
				return nil, fmt.Errorf("inference: KV cache layer %d state %q: %w", index, record.name, err)
			}
			bytes, ok := checked.Bytes(uint64(len(record.value.Data)), cacheScalarBytes)
			recordSize, okSize := checked.Add64(
				cacheRecordPrefixBytes, uint64(len(record.name)), cacheRecordValueBytes, bytes,
			)
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
	for _, name := range layer.States.SortedNames() {
		state := layer.States[name]
		records = append(records, cacheLayerRecord{name: string(name), mode: state.Mode, value: state.Value})
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
				stateName := model.CacheStateName(name)
				if err := validateLayerState(stateName, state, tokens); err != nil {
					return nil, fmt.Errorf("inference: KV cache layer %d state %q: %w", index, name, err)
				}
				if layer.States == nil {
					layer.States = make(LayerStates)
				}
				layer.States[stateName] = state
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

func validateLayerState(name model.CacheStateName, state LayerState, tokens uint32) error {
	if !validCacheStateName(string(name)) || name == "key" || name == "value" {
		return errors.New("invalid name")
	}
	if !state.Mode.Valid() {
		return fmt.Errorf("invalid mode %d", state.Mode)
	}
	if err := validateStateValue(state.Value); err != nil {
		return err
	}
	if state.Mode.TokenAligned() &&
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
