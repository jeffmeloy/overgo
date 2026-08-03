package model

import (
	"fmt"

	"llamacpp2go/internal/tensor"
)

// CacheExtent: state range behavior.
type CacheExtent uint8

const (
	CacheExtentFixed CacheExtent = iota
	CacheExtentToken
)

// CacheValueSchema: named or primary state contract.
type CacheValueSchema struct {
	Extent       CacheExtent
	Shape        tensor.Shape
	VariableLast bool
}

// LayerCacheSchema: layer cache contract.
type LayerCacheSchema struct {
	Label        string
	Primary      [2]CacheValueSchema
	States       map[string]CacheValueSchema
	StrictStates bool
}

// CacheSchema: derives layer cache shapes and range behavior.
func CacheSchema(spec Spec, layerIndex int, info LayerWeights, tokens uint32) (LayerCacheSchema, error) {
	schema := LayerCacheSchema{
		Label:  "KV",
		States: make(map[string]CacheValueSchema),
	}
	fixed := func(shape tensor.Shape) CacheValueSchema {
		return CacheValueSchema{Extent: CacheExtentFixed, Shape: shape}
	}
	token := func(shape tensor.Shape) CacheValueSchema {
		return CacheValueSchema{Extent: CacheExtentToken, Shape: shape}
	}
	if spec.Profile().Attention == AttentionDSA && spec.LayerHasFullIndexer(uint32(layerIndex)) {
		schema.States["indexer_key"] = token(tensor.MustShape(
			uint64(spec.IndexerKeyLength), 1, uint64(tokens),
		))
	}
	if spec.Architecture == "deepseek4" {
		ratio := spec.CompressRatios[layerIndex]
		schema.StrictStates = true
		schema.States["positions"] = token(tensor.MustShape(1, 1, uint64(tokens)))
		if ratio != 0 {
			coefficient := uint64(1)
			if ratio == 4 {
				coefficient = 2
			}
			shape := tensor.MustShape(coefficient*uint64(spec.KeyLength), 1, uint64(tokens))
			schema.States["compressor_kv"] = token(shape)
			schema.States["compressor_score"] = token(shape)
		}
		if ratio == 4 {
			shape := tensor.MustShape(2*uint64(spec.IndexerKeyLength), 1, uint64(tokens))
			schema.States["indexer_compressor_kv"] = token(shape)
			schema.States["indexer_compressor_score"] = token(shape)
		}
	}
	if spec.Architecture == "falcon-h1" {
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		schema.States["conv_state"] = fixed(tensor.MustShape(
			uint64(spec.SSMConvKernel-1), channels,
		))
		schema.States["ssm_state"] = fixed(tensor.MustShape(
			uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize),
		))
	}
	if spec.Architecture == "t5" {
		key := fixed(tensor.MustShape(uint64(spec.KeyLength), uint64(spec.HeadCountKV), 1))
		key.VariableLast = true
		value := fixed(tensor.MustShape(uint64(spec.ValueLength), uint64(spec.HeadCountKV), 1))
		value.VariableLast = true
		schema.States["cross_key"] = key
		schema.States["cross_value"] = value
	}

	first, second, recurrent, label, err := recurrentCacheSchema(spec, layerIndex, info)
	if err != nil {
		return LayerCacheSchema{}, err
	}
	if recurrent {
		schema.Label = label
		schema.Primary = [2]CacheValueSchema{fixed(first), fixed(second)}
		return schema, nil
	}
	if (spec.Architecture == "nemotron_h" || spec.Architecture == "nemotron_h_moe") &&
		spec.LayerFeedForwardLength(uint32(layerIndex)) > 0 ||
		spec.Architecture == "deci" && spec.LayerKVHeadCount(uint32(layerIndex)) == 0 {
		shape := tensor.MustShape(1, 1, uint64(tokens))
		schema.Label = "sentinel"
		schema.Primary = [2]CacheValueSchema{token(shape), token(shape)}
		return schema, nil
	}
	keyWidth := uint64(spec.LayerKeyLength(uint32(layerIndex)))
	valueWidth := uint64(spec.LayerValueLength(uint32(layerIndex)))
	heads := uint64(spec.LayerKVHeadCount(uint32(layerIndex)))
	attention := spec.Profile().Attention
	if attention == AttentionMLA || attention == AttentionDSA || spec.Architecture == "kimi-linear" {
		heads = uint64(spec.HeadCount)
		if info.AttentionKB != nil {
			keyWidth = uint64(spec.KVLoRARank + spec.RopeDimensionCount)
			valueWidth = uint64(spec.KVLoRARank)
			heads = 1
		}
	}
	schema.Primary = [2]CacheValueSchema{
		token(tensor.MustShape(keyWidth, heads, uint64(tokens))),
		token(tensor.MustShape(valueWidth, heads, uint64(tokens))),
	}
	return schema, nil
}

// PrimaryCacheExtent: primary state range behavior.
func PrimaryCacheExtent(spec Spec, layerIndex int, info LayerWeights) CacheExtent {
	if usesRecurrentPrimaryCache(spec, layerIndex, info) {
		return CacheExtentFixed
	}
	return CacheExtentToken
}

func usesRecurrentPrimaryCache(spec Spec, layerIndex int, info LayerWeights) bool {
	recurrent := info.Recurrent
	switch spec.Architecture {
	case "mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7":
		recurrent = true
	case "falcon-h1":
		recurrent = false
	case "nemotron_h", "nemotron_h_moe":
		recurrent = spec.IsRecurrentLayer(uint32(layerIndex))
	}
	if spec.Profile().Attention == AttentionQwenGDN {
		recurrent = spec.IsRecurrentLayer(uint32(layerIndex)) || info.Recurrent
	}
	return recurrent
}

func recurrentCacheSchema(
	spec Spec,
	layerIndex int,
	info LayerWeights,
) (tensor.Shape, tensor.Shape, bool, string, error) {
	if !usesRecurrentPrimaryCache(spec, layerIndex, info) {
		return tensor.Shape{}, tensor.Shape{}, false, "", nil
	}
	embedding := uint64(spec.EmbeddingLength)
	switch spec.Architecture {
	case "rwkv6":
		return tensor.MustShape(embedding, 2), tensor.MustShape(
			uint64(spec.WKVHeadSize), uint64(spec.WKVHeadSize), uint64(spec.HeadCount), 1,
		), true, "RWKV6", nil
	case "rwkv6qwen2":
		return tensor.MustShape(embedding), tensor.MustShape(
			uint64(spec.WKVHeadSize), uint64(spec.WKVHeadSize), uint64(spec.HeadCount), 1,
		), true, "RWKV6-Qwen2", nil
	case "rwkv7", "arwkv7":
		return tensor.MustShape(embedding, uint64(spec.TokenShiftCount)), tensor.MustShape(
			uint64(spec.WKVHeadSize), uint64(spec.WKVHeadSize), uint64(spec.HeadCount), 1,
		), true, "RWKV7", nil
	case "kimi-linear":
		return tensor.MustShape(uint64(spec.SSMConvKernel-1), 3*uint64(spec.SSMInnerSize)),
			tensor.MustShape(
				uint64(spec.KDAHeadDim), uint64(spec.KDAHeadDim), uint64(spec.HeadCount), 1,
			), true, "Kimi Linear recurrent", nil
	case "qwen3next", "qwen35", "qwen35moe":
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMStateSize)*uint64(spec.SSMGroupCount)
		return tensor.MustShape(uint64(spec.SSMConvKernel-1), channels), tensor.MustShape(
			uint64(spec.SSMStateSize), uint64(spec.SSMStateSize),
			uint64(spec.SSMTimeStepRank), 1,
		), true, "GDN recurrent", nil
	case "mamba2", "granitehybrid", "nemotron_h", "nemotron_h_moe":
		channels := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		return tensor.MustShape(uint64(spec.SSMConvKernel-1), channels),
			tensor.MustShape(uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)), true, "Mamba recurrent", nil
	case "mamba", "jamba", "plamo2":
		return tensor.MustShape(uint64(spec.SSMConvKernel-1), uint64(spec.SSMInnerSize)),
			tensor.MustShape(uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)), true, "Mamba recurrent", nil
	case "lfm2", "lfm2moe":
		return tensor.MustShape(
			uint64(spec.ShortConvCacheLength-1), uint64(spec.EmbeddingLength),
		), tensor.MustShape(1), true, "LFM2 recurrent", nil
	default:
		return tensor.Shape{}, tensor.Shape{}, false, "", fmt.Errorf(
			"architecture %s layer %d has no recurrent cache schema",
			spec.Architecture, layerIndex,
		)
	}
}
