package model

import (
	"fmt"
	"maps"
	"slices"

	"overgo/internal/tensor"
)

// CacheStateMode: serialized range behavior.
type CacheStateMode uint32

const (
	CacheStateFixed CacheStateMode = 1
	CacheStateToken CacheStateMode = 2
)

// CacheState: range behavior plus representation-specific value.
type CacheState[T any] struct {
	Mode  CacheStateMode
	Value T
}

// CacheStates: named representation-specific state collection.
type CacheStates[T any] map[CacheStateName]CacheState[T]

func (s CacheStates[T]) Clone() CacheStates[T] {
	return maps.Clone(s)
}

func (s CacheStates[T]) CloneValues(clone func(T) T) CacheStates[T] {
	return MapCacheStateValues(s, clone)
}

func (s CacheStates[T]) SortedNames() []CacheStateName {
	names := make([]CacheStateName, 0, len(s))
	for name := range s {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

func (s CacheStates[T]) AppendValues(values []T) []T {
	for _, name := range s.SortedNames() {
		values = append(values, s[name].Value)
	}
	return values
}

func MapCacheStateValues[T, U any](states CacheStates[T], transform func(T) U) CacheStates[U] {
	if states == nil {
		return nil
	}
	result := make(CacheStates[U], len(states))
	for name, state := range states {
		result[name] = CacheState[U]{Mode: state.Mode, Value: transform(state.Value)}
	}
	return result
}

func (m CacheStateMode) Valid() bool {
	return m == CacheStateFixed || m == CacheStateToken
}

func (m CacheStateMode) TokenAligned() bool {
	return m == CacheStateToken
}

// CacheStateName: serialized named-state ABI key.
type CacheStateName string

const (
	CacheStateIndexerKey             CacheStateName = "indexer_key"
	CacheStatePositions              CacheStateName = "positions"
	CacheStateCompressorKV           CacheStateName = "compressor_kv"
	CacheStateCompressorScore        CacheStateName = "compressor_score"
	CacheStateIndexerCompressorKV    CacheStateName = "indexer_compressor_kv"
	CacheStateIndexerCompressorScore CacheStateName = "indexer_compressor_score"
	CacheStateConvolution            CacheStateName = "conv_state"
	CacheStateSSM                    CacheStateName = "ssm_state"
	CacheStateCrossKey               CacheStateName = "cross_key"
	CacheStateCrossValue             CacheStateName = "cross_value"
)

// CacheValueSchema: named or primary state contract.
type CacheValueSchema struct {
	Shape        tensor.Shape
	VariableLast bool
	ZeroInitial  bool
}

// CachePair: primary key/value contract.
type CachePair[T any] struct {
	Key   T
	Value T
}

// NewCachePair: named pair construction.
func NewCachePair[T any](key, value T) CachePair[T] {
	return CachePair[T]{Key: key, Value: value}
}

// LayerCacheSchema: layer cache contract.
type LayerCacheSchema struct {
	Label        string
	Primary      CachePair[CacheState[CacheValueSchema]]
	StrictStates bool
	states       []namedCacheStateSchema
	tokens       uint32
}

type namedCacheStateSchema struct {
	name  CacheStateName
	state CacheState[CacheValueSchema]
}

func compileCacheSchemas(
	spec Spec,
	plans []LayerPlan,
	layers []LayerWeights,
) ([]LayerCacheSchema, error) {
	if len(layers) != 0 && len(plans) != len(layers) {
		return nil, fmt.Errorf(
			"cache schema layer count %d differs from plan count %d",
			len(layers), len(plans),
		)
	}
	result := make([]LayerCacheSchema, len(plans))
	for index := range plans {
		layer := LayerWeights{}
		if index < len(layers) {
			layer = layers[index]
		}
		schema, err := cacheSchemaForPlan(spec, plans[index], layer, 1)
		if err != nil {
			return nil, fmt.Errorf("cache schema layer %d: %w", index, err)
		}
		result[index] = schema
	}
	return result, nil
}

// WithTokenCount binds token-aligned dimensions without copying state templates.
func (s LayerCacheSchema) WithTokenCount(tokens uint32) LayerCacheSchema {
	s.tokens = tokens
	s.Primary.Key = s.materialize(s.Primary.Key)
	s.Primary.Value = s.materialize(s.Primary.Value)
	return s
}

func (s LayerCacheSchema) materialize(state CacheState[CacheValueSchema]) CacheState[CacheValueSchema] {
	if !state.Mode.TokenAligned() || state.Value.Shape.Rank == 0 {
		return state
	}
	tokens := s.tokens
	if tokens == 0 {
		tokens = 1
	}
	state.Value.Shape.Dims[state.Value.Shape.Rank-1] = uint64(tokens)
	return state
}

// State returns one materialized named-state contract.
func (s LayerCacheSchema) State(name CacheStateName) (CacheState[CacheValueSchema], bool) {
	for _, candidate := range s.states {
		if candidate.name == name {
			return s.materialize(candidate.state), true
		}
	}
	return CacheState[CacheValueSchema]{}, false
}

// RangeStates visits materialized named-state contracts in stable order.
func (s LayerCacheSchema) RangeStates(visit func(CacheStateName, CacheState[CacheValueSchema])) {
	for _, candidate := range s.states {
		visit(candidate.name, s.materialize(candidate.state))
	}
}

func (s LayerCacheSchema) HasState(name CacheStateName) bool {
	_, present := s.State(name)
	return present
}

func (s LayerCacheSchema) StateCount() int {
	return len(s.states)
}

func (s *LayerCacheSchema) addState(name CacheStateName, state CacheState[CacheValueSchema]) {
	s.states = append(s.states, namedCacheStateSchema{name: name, state: state})
}

type cacheSchemaBuilder struct {
	schema *LayerCacheSchema
	err    error
}

func (b *cacheSchemaBuilder) state(
	name CacheStateName,
	mode CacheStateMode,
	zeroInitial, variableLast bool,
	dimensions ...uint64,
) {
	if b.err != nil {
		return
	}
	shape, err := cacheShape(dimensions...)
	if err != nil {
		b.err = fmt.Errorf("state %q: %w", name, err)
		return
	}
	b.schema.addState(name, CacheState[CacheValueSchema]{
		Mode: mode,
		Value: CacheValueSchema{
			Shape: shape, ZeroInitial: zeroInitial, VariableLast: variableLast,
		},
	})
}

func cacheValue(mode CacheStateMode, shape tensor.Shape) CacheState[CacheValueSchema] {
	return CacheState[CacheValueSchema]{Mode: mode, Value: CacheValueSchema{Shape: shape}}
}

func cacheSchemaForPlan(
	spec Spec,
	plan LayerPlan,
	info LayerWeights,
	tokens uint32,
) (LayerCacheSchema, error) {
	layerIndex := int(plan.Layer)
	schema := LayerCacheSchema{Label: "KV"}
	states := cacheSchemaBuilder{schema: &schema}
	if plan.Attention == AttentionDSA && spec.LayerHasFullIndexer(plan.Layer) {
		states.state(CacheStateIndexerKey, CacheStateToken, false, false,
			uint64(spec.IndexerKeyLength), 1, 1)
	}
	if plan.Cache == CacheDeepSeek4 {
		if layerIndex < 0 || layerIndex >= len(spec.CompressRatios) {
			return LayerCacheSchema{}, fmt.Errorf(
				"DeepSeek4 compression ratio for layer %d is unavailable", layerIndex,
			)
		}
		ratio := tensor.DeepSeek4CompressionRatio(spec.CompressRatios[layerIndex])
		schema.StrictStates = true
		states.state(CacheStatePositions, CacheStateToken, false, false, 1, 1, 1)
		if ratio.Enabled() {
			coefficient := ratio.KVWidthMultiplier()
			width := coefficient * uint64(spec.KeyLength)
			states.state(CacheStateCompressorKV, CacheStateToken, false, false, width, 1, 1)
			states.state(CacheStateCompressorScore, CacheStateToken, false, false, width, 1, 1)
		}
		if ratio.UsesIndexer() {
			width := 2 * uint64(spec.IndexerKeyLength)
			states.state(CacheStateIndexerCompressorKV, CacheStateToken, false, false, width, 1, 1)
			states.state(CacheStateIndexerCompressorScore, CacheStateToken, false, false, width, 1, 1)
		}
	}
	if plan.Cache == CacheFalconH1 {
		if spec.SSMConvKernel == 0 {
			return LayerCacheSchema{}, fmt.Errorf("Falcon-H1 convolution kernel is zero")
		}
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		states.state(CacheStateConvolution, CacheStateFixed, true, false,
			uint64(spec.SSMConvKernel-1), channels)
		states.state(CacheStateSSM, CacheStateFixed, true, false,
			uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize))
	}
	if plan.Cache == CacheT5 {
		shapes := spec.TensorShapes(plan.Layer)
		states.state(CacheStateCrossKey, CacheStateFixed, false, true,
			shapes.Key, shapes.KVHeads, 1)
		states.state(CacheStateCrossValue, CacheStateFixed, false, true,
			shapes.Value, shapes.KVHeads, 1)
	}
	if states.err != nil {
		return LayerCacheSchema{}, states.err
	}

	first, second, recurrent, label, err := recurrentCacheSchema(spec, plan)
	if err != nil {
		return LayerCacheSchema{}, err
	}
	if recurrent {
		schema.Label = label
		schema.Primary = NewCachePair(
			cacheValue(CacheStateFixed, first), cacheValue(CacheStateFixed, second),
		)
		return schema.WithTokenCount(tokens), nil
	}
	if plan.Cache == CacheSentinel {
		shape, err := cacheShape(1, 1, 1)
		if err != nil {
			return LayerCacheSchema{}, err
		}
		schema.Label = "sentinel"
		state := cacheValue(CacheStateToken, shape)
		schema.Primary = NewCachePair(state, state)
		return schema.WithTokenCount(tokens), nil
	}
	shapes := spec.TensorShapes(uint32(layerIndex))
	keyWidth, valueWidth, heads := shapes.Key, shapes.Value, shapes.KVHeads
	if plan.Attention == AttentionMLA || plan.Attention == AttentionDSA || plan.Block == BlockKimiLinear {
		heads = uint64(spec.HeadCount)
		if info.AttentionKB != nil {
			keyWidth = uint64(spec.KVLoRARank + spec.RopeDimensionCount)
			valueWidth = uint64(spec.KVLoRARank)
			heads = 1
		}
	}
	shapes.Key, shapes.Value, shapes.KVHeads = keyWidth, valueWidth, heads
	key, err := shapes.KeyCacheShape(1)
	if err != nil {
		return LayerCacheSchema{}, fmt.Errorf("key cache: %w", err)
	}
	value, err := shapes.ValueCacheShape(1)
	if err != nil {
		return LayerCacheSchema{}, fmt.Errorf("value cache: %w", err)
	}
	schema.Primary = NewCachePair(
		cacheValue(CacheStateToken, key), cacheValue(CacheStateToken, value),
	)
	return schema.WithTokenCount(tokens), nil
}

func cacheShape(dimensions ...uint64) (tensor.Shape, error) {
	shape, err := tensor.NewShape(dimensions...)
	if err != nil {
		return tensor.Shape{}, fmt.Errorf("cache shape %v: %w", dimensions, err)
	}
	return shape, nil
}

func recurrentCacheSchema(
	spec Spec,
	plan LayerPlan,
) (tensor.Shape, tensor.Shape, bool, string, error) {
	if plan.CacheMode != CacheStateFixed {
		return tensor.Shape{}, tensor.Shape{}, false, "", nil
	}
	embedding := uint64(spec.EmbeddingLength)
	stateMatrix := []uint64{
		uint64(spec.WKVHeadSize), uint64(spec.WKVHeadSize), uint64(spec.HeadCount), 1,
	}
	switch plan.Cache {
	case CacheRWKV6:
		return recurrentCachePair("RWKV6", []uint64{embedding, 2}, stateMatrix)
	case CacheRWKV6Qwen2:
		return recurrentCachePair("RWKV6-Qwen2", []uint64{embedding}, stateMatrix)
	case CacheRWKV7:
		return recurrentCachePair(
			"RWKV7", []uint64{embedding, uint64(spec.TokenShiftCount)}, stateMatrix,
		)
	case CacheKimiLinear:
		previous, err := previousCacheElements(spec.SSMConvKernel, "Kimi Linear convolution kernel")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		return recurrentCachePair(
			"Kimi Linear recurrent",
			[]uint64{previous, 3 * uint64(spec.SSMInnerSize)},
			[]uint64{uint64(spec.KDAHeadDim), uint64(spec.KDAHeadDim), uint64(spec.HeadCount), 1},
		)
	case CacheQwenGDN:
		previous, err := previousCacheElements(spec.SSMConvKernel, "GDN convolution kernel")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMStateSize)*uint64(spec.SSMGroupCount)
		return recurrentCachePair(
			"GDN recurrent", []uint64{previous, channels},
			[]uint64{
				uint64(spec.SSMStateSize), uint64(spec.SSMStateSize),
				uint64(spec.SSMTimeStepRank), 1,
			},
		)
	case CacheMamba2:
		previous, err := previousCacheElements(spec.SSMConvKernel, "Mamba2 convolution kernel")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		return recurrentCachePair(
			"Mamba recurrent", []uint64{previous, channels},
			[]uint64{uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)},
		)
	case CacheMamba:
		previous, err := previousCacheElements(spec.SSMConvKernel, "Mamba convolution kernel")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		return recurrentCachePair(
			"Mamba recurrent", []uint64{previous, uint64(spec.SSMInnerSize)},
			[]uint64{uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)},
		)
	case CacheLFM2:
		previous, err := previousCacheElements(spec.ShortConvCacheLength, "LFM2 convolution cache length")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		return recurrentCachePair(
			"LFM2 recurrent", []uint64{previous, uint64(spec.EmbeddingLength)}, []uint64{1},
		)
	default:
		return tensor.Shape{}, tensor.Shape{}, false, "", fmt.Errorf(
			"architecture %s layer %d has no recurrent cache schema",
			spec.Architecture, plan.Layer,
		)
	}
}

func recurrentCachePair(
	label string,
	firstDimensions, secondDimensions []uint64,
) (tensor.Shape, tensor.Shape, bool, string, error) {
	first, err := cacheShape(firstDimensions...)
	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, false, "", fmt.Errorf("%s first state: %w", label, err)
	}
	second, err := cacheShape(secondDimensions...)
	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, false, "", fmt.Errorf("%s second state: %w", label, err)
	}
	return first, second, true, label, nil
}

func previousCacheElements(length uint32, label string) (uint64, error) {
	if length <= 1 {
		return 0, fmt.Errorf("%s %d leaves no cache history", label, length)
	}
	return uint64(length - 1), nil
}
