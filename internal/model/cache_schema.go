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
	States       CacheStates[CacheValueSchema]
	StrictStates bool
}

// CacheSchema: derives layer cache shapes and range behavior.
func CacheSchema(spec Spec, layerIndex int, info LayerWeights, tokens uint32) (LayerCacheSchema, error) {
	plan := spec.PlanLayer(uint32(layerIndex), info.Recurrent)
	return CacheSchemaForPlan(spec, plan, info, tokens)
}

// CacheSchemaForPlan: materializes a compiled layer-state contract.
func CacheSchemaForPlan(
	spec Spec,
	plan LayerPlan,
	info LayerWeights,
	tokens uint32,
) (LayerCacheSchema, error) {
	layerIndex := int(plan.Layer)
	shapeTokens := tokens
	if shapeTokens == 0 {
		shapeTokens = 1
	}
	schema := LayerCacheSchema{
		Label:  "KV",
		States: make(CacheStates[CacheValueSchema]),
	}
	fixed := func(shape tensor.Shape) CacheState[CacheValueSchema] {
		return CacheState[CacheValueSchema]{
			Mode: CacheStateFixed, Value: CacheValueSchema{Shape: shape},
		}
	}
	fixedZero := func(shape tensor.Shape) CacheState[CacheValueSchema] {
		state := fixed(shape)
		state.Value.ZeroInitial = true
		return state
	}
	token := func(shape tensor.Shape) CacheState[CacheValueSchema] {
		return CacheState[CacheValueSchema]{
			Mode: CacheStateToken, Value: CacheValueSchema{Shape: shape},
		}
	}
	if plan.Attention == AttentionDSA && spec.LayerHasFullIndexer(plan.Layer) {
		schema.States[CacheStateIndexerKey] = token(tensor.MustShape(
			uint64(spec.IndexerKeyLength), 1, uint64(shapeTokens),
		))
	}
	if plan.Cache == CacheDeepSeek4 {
		ratio := tensor.DeepSeek4CompressionRatio(spec.CompressRatios[layerIndex])
		schema.StrictStates = true
		schema.States[CacheStatePositions] = token(tensor.MustShape(1, 1, uint64(shapeTokens)))
		if ratio.Enabled() {
			coefficient := ratio.KVWidthMultiplier()
			shape := tensor.MustShape(coefficient*uint64(spec.KeyLength), 1, uint64(shapeTokens))
			schema.States[CacheStateCompressorKV] = token(shape)
			schema.States[CacheStateCompressorScore] = token(shape)
		}
		if ratio.UsesIndexer() {
			shape := tensor.MustShape(2*uint64(spec.IndexerKeyLength), 1, uint64(shapeTokens))
			schema.States[CacheStateIndexerCompressorKV] = token(shape)
			schema.States[CacheStateIndexerCompressorScore] = token(shape)
		}
	}
	if plan.Cache == CacheFalconH1 {
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		schema.States[CacheStateConvolution] = fixedZero(tensor.MustShape(
			uint64(spec.SSMConvKernel-1), channels,
		))
		schema.States[CacheStateSSM] = fixedZero(tensor.MustShape(
			uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize),
		))
	}
	if plan.Cache == CacheT5 {
		shapes := spec.TensorShapes(plan.Layer)
		key := fixed(shapes.KeyCache(1))
		key.Value.VariableLast = true
		value := fixed(shapes.ValueCache(1))
		value.Value.VariableLast = true
		schema.States[CacheStateCrossKey] = key
		schema.States[CacheStateCrossValue] = value
	}

	first, second, recurrent, label, err := recurrentCacheSchema(spec, plan)
	if err != nil {
		return LayerCacheSchema{}, err
	}
	if recurrent {
		schema.Label = label
		schema.Primary = NewCachePair(fixed(first), fixed(second))
		return schema, nil
	}
	if plan.Cache == CacheSentinel {
		shape := tensor.MustShape(1, 1, uint64(shapeTokens))
		schema.Label = "sentinel"
		schema.Primary = NewCachePair(token(shape), token(shape))
		return schema, nil
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
	schema.Primary = NewCachePair(
		token(shapes.KeyCache(shapeTokens)), token(shapes.ValueCache(shapeTokens)),
	)
	return schema, nil
}

func recurrentCacheSchema(
	spec Spec,
	plan LayerPlan,
) (tensor.Shape, tensor.Shape, bool, string, error) {
	if plan.CacheMode != CacheStateFixed {
		return tensor.Shape{}, tensor.Shape{}, false, "", nil
	}
	embedding := uint64(spec.EmbeddingLength)
	switch plan.Cache {
	case CacheRWKV6:
		return tensor.MustShape(embedding, 2), tensor.MustShape(
			uint64(spec.WKVHeadSize), uint64(spec.WKVHeadSize), uint64(spec.HeadCount), 1,
		), true, "RWKV6", nil
	case CacheRWKV6Qwen2:
		return tensor.MustShape(embedding), tensor.MustShape(
			uint64(spec.WKVHeadSize), uint64(spec.WKVHeadSize), uint64(spec.HeadCount), 1,
		), true, "RWKV6-Qwen2", nil
	case CacheRWKV7:
		return tensor.MustShape(embedding, uint64(spec.TokenShiftCount)), tensor.MustShape(
			uint64(spec.WKVHeadSize), uint64(spec.WKVHeadSize), uint64(spec.HeadCount), 1,
		), true, "RWKV7", nil
	case CacheKimiLinear:
		return tensor.MustShape(uint64(spec.SSMConvKernel-1), 3*uint64(spec.SSMInnerSize)),
			tensor.MustShape(
				uint64(spec.KDAHeadDim), uint64(spec.KDAHeadDim), uint64(spec.HeadCount), 1,
			), true, "Kimi Linear recurrent", nil
	case CacheQwenGDN:
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMStateSize)*uint64(spec.SSMGroupCount)
		return tensor.MustShape(uint64(spec.SSMConvKernel-1), channels), tensor.MustShape(
			uint64(spec.SSMStateSize), uint64(spec.SSMStateSize),
			uint64(spec.SSMTimeStepRank), 1,
		), true, "GDN recurrent", nil
	case CacheMamba2:
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		return tensor.MustShape(uint64(spec.SSMConvKernel-1), channels),
			tensor.MustShape(uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)), true, "Mamba recurrent", nil
	case CacheMamba:
		return tensor.MustShape(uint64(spec.SSMConvKernel-1), uint64(spec.SSMInnerSize)),
			tensor.MustShape(uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)), true, "Mamba recurrent", nil
	case CacheLFM2:
		return tensor.MustShape(
			uint64(spec.ShortConvCacheLength-1), uint64(spec.EmbeddingLength),
		), tensor.MustShape(1), true, "LFM2 recurrent", nil
	default:
		return tensor.Shape{}, tensor.Shape{}, false, "", fmt.Errorf(
			"architecture %s layer %d has no recurrent cache schema",
			spec.Architecture, plan.Layer,
		)
	}
}
