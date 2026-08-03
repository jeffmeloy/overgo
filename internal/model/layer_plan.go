package model

import "fmt"

// BlockPolicy: compiled block-builder selection.
type BlockPolicy uint8

const (
	BlockDense BlockPolicy = iota
	BlockMamba
	BlockMamba2
	BlockFalconH1
	BlockJamba
	BlockGraniteHybrid
	BlockPLaMo2
	BlockNemotronH
	BlockKimiLinear
	BlockMLA
	BlockDSA
	BlockDeepSeek4
)

// CachePolicy: compiled layer-state layout.
type CachePolicy uint8

const (
	CacheAttention CachePolicy = iota
	CacheSentinel
	CacheMamba
	CacheMamba2
	CacheRWKV6
	CacheRWKV6Qwen2
	CacheRWKV7
	CacheKimiLinear
	CacheQwenGDN
	CacheLFM2
	CacheFalconH1
	CacheT5
	CacheDeepSeek4
)

// LayerPlan: derived layer execution contract.
type LayerPlan struct {
	Layer         uint32
	GraphFamily   ArchitectureFamily
	CatalogFamily ArchitectureFamily
	Block         BlockPolicy
	Cache         CachePolicy
	Attention     AttentionPolicy
	Position      PositionPolicy
	Residual      ResidualPolicy
	FeedForward   FeedForwardPolicy
	CacheExtent   CacheExtent
	Recurrent     bool
	Sliding       bool
	UsesRoPE      bool
	MultiAxis     bool
	HasKV         bool
}

// PlanLayer: derives graph and cache behavior once per layer.
func (s Spec) PlanLayer(layer uint32, recurrent bool) LayerPlan {
	profile := s.Profile()
	info := LayerWeights{Recurrent: recurrent}
	recurrent = recurrent || s.IsRecurrentLayer(layer)
	return LayerPlan{
		Layer:         layer,
		GraphFamily:   profile.GraphFamily,
		CatalogFamily: profile.CatalogFamily,
		Attention:     profile.Attention,
		Position:      profile.Position,
		Residual:      profile.Residual,
		FeedForward:   profile.FeedForward,
		Block:         blockPolicy(s, profile, recurrent),
		Cache:         cachePolicy(s, profile, layer, recurrent),
		CacheExtent:   PrimaryCacheExtent(s, int(layer), info),
		Recurrent:     recurrent,
		Sliding:       s.IsSlidingLayer(layer),
		UsesRoPE:      s.UsesRoPE(layer),
		MultiAxis:     profile.Has(ArchitectureMultiAxisPositions),
		HasKV:         s.LayerHasKV(layer),
	}
}

// ModelPlan: immutable model and per-layer execution contract.
type ModelPlan struct {
	Profile ArchitectureProfile
	Layers  []LayerPlan
}

// CompileModelPlan: resolves architecture decisions before execution.
func CompileModelPlan(spec Spec, weights Weights) (ModelPlan, error) {
	profile, ok := LookupArchitecture(spec.Architecture)
	if !ok {
		return ModelPlan{}, &UnsupportedArchitectureError{Architecture: spec.Architecture}
	}
	layers := spec.BlockCount
	if profile.Family == ArchitectureFamilyEncoderDecoder && spec.DecoderBlockCount > layers {
		layers = spec.DecoderBlockCount
	}
	plan := ModelPlan{Profile: profile, Layers: make([]LayerPlan, layers)}
	for layer := range layers {
		recurrent := int(layer) < len(weights.Layers) && weights.Layers[layer].Recurrent
		plan.Layers[layer] = spec.PlanLayer(layer, recurrent)
	}
	return plan, nil
}

// Layer: bounds-checked layer contract.
func (p ModelPlan) Layer(layer int) (LayerPlan, error) {
	if layer < 0 || layer >= len(p.Layers) {
		return LayerPlan{}, fmt.Errorf("model plan layer %d is outside [0,%d)", layer, len(p.Layers))
	}
	return p.Layers[layer], nil
}

func blockPolicy(spec Spec, profile ArchitectureProfile, recurrent bool) BlockPolicy {
	switch {
	case profile.Attention == AttentionDSA:
		return BlockDSA
	case spec.Architecture == "deepseek4":
		return BlockDeepSeek4
	case profile.Attention == AttentionMLA:
		return BlockMLA
	}
	switch spec.Architecture {
	case "mamba":
		return BlockMamba
	case "mamba2":
		return BlockMamba2
	case "falcon-h1":
		return BlockFalconH1
	case "jamba":
		if recurrent {
			return BlockJamba
		}
	case "granitehybrid":
		if recurrent {
			return BlockGraniteHybrid
		}
	case "plamo2":
		if recurrent {
			return BlockPLaMo2
		}
	case "nemotron_h", "nemotron_h_moe":
		return BlockNemotronH
	case "kimi-linear":
		return BlockKimiLinear
	}
	return BlockDense
}

func cachePolicy(spec Spec, profile ArchitectureProfile, layer uint32, recurrent bool) CachePolicy {
	switch spec.Architecture {
	case "deepseek4":
		return CacheDeepSeek4
	case "falcon-h1":
		return CacheFalconH1
	case "t5":
		return CacheT5
	case "rwkv6":
		return CacheRWKV6
	case "rwkv6qwen2":
		return CacheRWKV6Qwen2
	case "rwkv7", "arwkv7":
		return CacheRWKV7
	case "kimi-linear":
		if recurrent {
			return CacheKimiLinear
		}
	case "qwen3next", "qwen35", "qwen35moe":
		if recurrent {
			return CacheQwenGDN
		}
	case "mamba2":
		return CacheMamba2
	case "granitehybrid", "nemotron_h", "nemotron_h_moe":
		if recurrent {
			return CacheMamba2
		}
	case "mamba":
		return CacheMamba
	case "jamba", "plamo2":
		if recurrent {
			return CacheMamba
		}
	case "lfm2", "lfm2moe":
		if recurrent {
			return CacheLFM2
		}
	}
	if (spec.Architecture == "nemotron_h" || spec.Architecture == "nemotron_h_moe") &&
		spec.LayerFeedForwardLength(layer) > 0 ||
		spec.Architecture == "deci" && spec.LayerKVHeadCount(layer) == 0 {
		return CacheSentinel
	}
	_ = profile
	return CacheAttention
}
