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
	CacheStateFixed CacheStateMode = iota + tensor.SingletonExtent
	CacheStateToken
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

func (m CacheStateMode) MatchesTokenExtent(shape tensor.Shape, tokens uint32) bool {
	if !m.TokenAligned() {
		return true
	}
	extent, _, valid := tensor.FinalExtent(shape)
	return valid && extent == uint64(tokens)
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

func (s CacheValueSchema) Matches(shape tensor.Shape) bool {
	if !s.VariableLast {
		return shape.Equal(s.Shape)
	}
	if shape.Rank != s.Shape.Rank || shape.Rank == tensor.FirstOffset {
		return false
	}
	last := int(shape.Rank) - tensor.SingletonExtent
	for index := range last {
		if shape.Dims[index] != s.Shape.Dims[index] {
			return false
		}
	}
	return shape.Dims[last] > tensor.FirstOffset
}

func (s CacheValueSchema) MatchesTrailingExtent(shape tensor.Shape, extent uint64) bool {
	expected, valid := tensor.WithTrailingExtent(s.Shape, extent)
	return valid && shape.Equal(expected)
}

func (s CacheValueSchema) AcceptsTokenTarget(mode CacheStateMode, shape tensor.Shape) bool {
	s.VariableLast = true
	return mode.TokenAligned() && s.Matches(shape)
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
		schema, err := cacheSchemaForPlan(spec, plans[index], layer, tensor.SingletonExtent)
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
	if !state.Mode.TokenAligned() {
		return state
	}
	tokens := s.tokens
	if tokens == tensor.FirstOffset {
		tokens = tensor.SingletonExtent
	}
	shape, valid := tensor.WithTrailingExtent(state.Value.Shape, uint64(tokens))
	if !valid {
		return state
	}
	state.Value.Shape = shape
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
	shape, err := tensor.NewShape(dimensions...)
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
	schema := LayerCacheSchema{Label: "KV"}
	states := cacheSchemaBuilder{schema: &schema}
	if plan.Attention == AttentionSparseLatent && spec.LayerHasFullIndexer(plan.Layer) {
		states.state(CacheStateIndexerKey, CacheStateToken, false, false,
			uint64(spec.IndexerKeyLength), tensor.SingletonExtent, tensor.SingletonExtent)
	}
	if plan.Cache == CacheCompressedAttention {
		if int(plan.Layer) >= len(spec.CompressRatios) {
			return LayerCacheSchema{}, fmt.Errorf(
				"compressed-attention ratio for layer %d is unavailable", plan.Layer,
			)
		}
		ratio := tensor.CompressionRatio(spec.CompressRatios[plan.Layer])
		schema.StrictStates = true
		states.state(
			CacheStatePositions, CacheStateToken, false, false,
			tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent,
		)
		if ratio.Enabled() {
			coefficient := ratio.KVWidthMultiplier()
			width := coefficient * uint64(spec.KeyLength)
			states.state(CacheStateCompressorKV, CacheStateToken, false, false,
				width, tensor.SingletonExtent, tensor.SingletonExtent)
			states.state(CacheStateCompressorScore, CacheStateToken, false, false,
				width, tensor.SingletonExtent, tensor.SingletonExtent)
		}
		if ratio.UsesIndexer() {
			width := tensor.PairedExtent * uint64(spec.IndexerKeyLength)
			states.state(CacheStateIndexerCompressorKV, CacheStateToken, false, false,
				width, tensor.SingletonExtent, tensor.SingletonExtent)
			states.state(CacheStateIndexerCompressorScore, CacheStateToken, false, false,
				width, tensor.SingletonExtent, tensor.SingletonExtent)
		}
	}
	if plan.Cache == CacheHybridAttentionScan {
		if spec.ssmConvolutionWindow() == tensor.FirstOffset {
			return LayerCacheSchema{}, fmt.Errorf("hybrid attention-scan convolution kernel is zero")
		}
		channels := uint64(spec.SSMInnerSize) +
			tensor.PairedExtent*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		states.state(CacheStateConvolution, CacheStateFixed, true, false,
			spec.ssmConvolutionWindow(), channels)
		states.state(CacheStateSSM, CacheStateFixed, true, false,
			uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize))
	}
	if plan.Cache == CacheCrossAttention {
		shapes := spec.TensorShapes(plan.Layer)
		states.state(CacheStateCrossKey, CacheStateFixed, false, true,
			shapes.Key, shapes.KVHeads, tensor.SingletonExtent)
		states.state(CacheStateCrossValue, CacheStateFixed, false, true,
			shapes.Value, shapes.KVHeads, tensor.SingletonExtent)
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
		shape := tensor.MustShape(tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent)
		schema.Label = "sentinel"
		state := cacheValue(CacheStateToken, shape)
		schema.Primary = NewCachePair(state, state)
		return schema.WithTokenCount(tokens), nil
	}
	shapes := spec.TensorShapes(plan.Layer)
	keyWidth, valueWidth, heads := shapes.Key, shapes.Value, shapes.KVHeads
	if plan.Attention == AttentionLatent || plan.Attention == AttentionSparseLatent || plan.Mixer == recurrentMixerKeyedDelta {
		heads = uint64(spec.HeadCount)
		if info.AttentionKB != nil {
			keyWidth = uint64(spec.KVLoRARank + spec.RopeDimensionCount)
			valueWidth = uint64(spec.KVLoRARank)
			heads = tensor.SingletonExtent
		}
	}
	shapes.Key, shapes.Value, shapes.KVHeads = keyWidth, valueWidth, heads
	key, err := tensor.NewShape(shapes.Key, shapes.KVHeads, tensor.SingletonExtent)
	if err != nil {
		return LayerCacheSchema{}, fmt.Errorf("key cache: %w", err)
	}
	value, err := tensor.NewShape(shapes.Value, shapes.KVHeads, tensor.SingletonExtent)
	if err != nil {
		return LayerCacheSchema{}, fmt.Errorf("value cache: %w", err)
	}
	schema.Primary = NewCachePair(
		cacheValue(CacheStateToken, key), cacheValue(CacheStateToken, value),
	)
	return schema.WithTokenCount(tokens), nil
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
		uint64(spec.WKVHeadSize), uint64(spec.WKVHeadSize),
		uint64(spec.HeadCount), tensor.SingletonExtent,
	}
	switch plan.Cache {
	case CacheDoubleTokenShiftRecurrence:
		return recurrentCachePair(
			"double token-shift recurrence", []uint64{embedding, tensor.PairedExtent}, stateMatrix,
		)
	case CacheSingleTokenShiftRecurrence:
		return recurrentCachePair("single token-shift recurrence", []uint64{embedding}, stateMatrix)
	case CacheVariableTokenShiftRecurrence:
		return recurrentCachePair(
			"variable token-shift recurrence", []uint64{embedding, uint64(spec.TokenShiftCount)}, stateMatrix,
		)
	case CacheKeyedDelta:
		previous, err := previousCacheElements(spec.SSMConvKernel, "keyed-delta convolution kernel")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		return recurrentCachePair(
			"keyed-delta recurrence",
			[]uint64{previous, tensor.TripleExtent * uint64(spec.SSMInnerSize)},
			[]uint64{
				uint64(spec.KDAHeadDim), uint64(spec.KDAHeadDim),
				uint64(spec.HeadCount), tensor.SingletonExtent,
			},
		)
	case CacheGatedDelta:
		previous, err := previousCacheElements(spec.SSMConvKernel, "gated-delta convolution kernel")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		channels := uint64(spec.SSMInnerSize) +
			tensor.PairedExtent*uint64(spec.SSMStateSize)*uint64(spec.SSMGroupCount)
		return recurrentCachePair(
			"gated-delta recurrence", []uint64{previous, channels},
			[]uint64{
				uint64(spec.SSMStateSize), uint64(spec.SSMStateSize),
				uint64(spec.SSMTimeStepRank), tensor.SingletonExtent,
			},
		)
	case CacheGroupedSelectiveScan:
		previous, err := previousCacheElements(spec.SSMConvKernel, "grouped selective-scan convolution kernel")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		channels := uint64(spec.SSMInnerSize) +
			tensor.PairedExtent*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		return recurrentCachePair(
			"grouped selective-scan recurrence", []uint64{previous, channels},
			[]uint64{uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)},
		)
	case CacheSelectiveScan:
		previous, err := previousCacheElements(spec.SSMConvKernel, "selective-scan convolution kernel")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		return recurrentCachePair(
			"selective-scan recurrence", []uint64{previous, uint64(spec.SSMInnerSize)},
			[]uint64{uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)},
		)
	case CacheShortConvolution:
		previous, err := previousCacheElements(spec.ShortConvCacheLength, "short-convolution cache length")
		if err != nil {
			return tensor.Shape{}, tensor.Shape{}, false, "", err
		}
		return recurrentCachePair(
			"short-convolution recurrence", []uint64{previous, uint64(spec.EmbeddingLength)},
			[]uint64{tensor.SingletonExtent},
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
	first, err := tensor.NewShape(firstDimensions...)
	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, false, "", fmt.Errorf("%s first state: %w", label, err)
	}
	second, err := tensor.NewShape(secondDimensions...)
	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, false, "", fmt.Errorf("%s second state: %w", label, err)
	}
	return first, second, true, label, nil
}

func previousCacheElements(length uint32, label string) (uint64, error) {
	if length <= tensor.SingletonExtent {
		return tensor.FirstOffset, fmt.Errorf("%s %d leaves no cache history", label, length)
	}
	return uint64(length - tensor.SingletonExtent), nil
}
