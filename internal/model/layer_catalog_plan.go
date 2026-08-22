package model

import (
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type rotaryCatalogPlan struct {
	factor, factorFallback       string
	long, longFallback           string
	short, shortFallback         string
	factorElements, longElements uint64
	reject, require, selectLong  bool
}

func (s Spec) rotaryCatalogPlan(profile ArchitectureProfile, layer LayerPlan) rotaryCatalogPlan {
	prefix := fmt.Sprintf("blk.%d.", layer.Layer)
	plan := rotaryCatalogPlan{
		factor: prefix + "rope_freqs.weight", factorElements: uint64(s.KeyLength),
		reject: layer.Attention == AttentionGatedDelta,
	}
	if s.RopeDimensionCount > tensor.FirstOffset {
		plan.factorElements = uint64(s.RopeDimensionCount)
	}
	if profile.Rotary.FactorPairs ||
		profile.LayerTopology == LayerTopologySharedKVAdapter && !layer.Sliding {
		plan.factorFallback = "rope_freqs.weight"
	}
	if !profile.Has(ArchitectureLongRoPE) {
		return plan
	}
	plan.long, plan.short = prefix+"rope_factors_long.weight", prefix+"rope_factors_short.weight"
	if profile.Rotary.FactorPairs {
		plan.long, plan.short = "rope_factors_long.weight", "rope_factors_short.weight"
	} else if layer.Layer > tensor.FirstOffset {
		plan.longFallback = firstBlockTensorPrefix + "rope_factors_long.weight"
		plan.shortFallback = firstBlockTensorPrefix + "rope_factors_short.weight"
	}
	plan.longElements = uint64(s.RopeDimensionCount / rotaryPairAlignment)
	plan.require = s.RopeScalingType == ropeScalingLongRoPE
	plan.selectLong = s.ContextLength > s.OriginalContextLength
	return plan
}

func firstCatalogTensor(catalog weightCatalog, primary, fallback string) (gguf.TensorInfo, bool) {
	if item, found := catalog.tensor(primary); found {
		return item, true
	}
	if fallback != "" {
		return catalog.tensor(fallback)
	}
	return gguf.TensorInfo{}, false
}

func (p rotaryCatalogPlan) bind(catalog weightCatalog, architecture string, layer *LayerWeights) error {
	factors, found := firstCatalogTensor(catalog, p.factor, p.factorFallback)
	if found {
		if p.reject {
			return fmt.Errorf("tensor %q requires unsupported hybrid RoPE factors", factors.Name)
		}
		if p.factorElements%uint64(rotaryPairAlignment) != tensor.FirstOffset {
			return fmt.Errorf("tensor %q has odd rotary width %d", factors.Name, p.factorElements)
		}
		if err := validateTensorInfo(factors, []dtype.Type{dtype.F32},
			[]uint64{p.factorElements / uint64(rotaryPairAlignment)}); err != nil {
			return err
		}
		layer.RopeFactors = &factors
	}
	if p.long == "" {
		return nil
	}
	long, hasLong := firstCatalogTensor(catalog, p.long, p.longFallback)
	short, hasShort := firstCatalogTensor(catalog, p.short, p.shortFallback)
	if hasLong != hasShort {
		return fmt.Errorf("%s LongRoPE factor tensors must both be present or absent", architecture)
	}
	if p.require && !hasLong {
		return fmt.Errorf("%s LongRoPE factor tensors are missing", architecture)
	}
	if !hasLong {
		return nil
	}
	for _, item := range []gguf.TensorInfo{long, short} {
		if err := validateTensorInfo(item, []dtype.Type{dtype.F32}, []uint64{p.longElements}); err != nil {
			return err
		}
	}
	if p.selectLong {
		layer.RopeFactors = &long
	} else {
		layer.RopeFactors = &short
	}
	return nil
}
