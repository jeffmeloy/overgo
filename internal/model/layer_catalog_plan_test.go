package model

import (
	"testing"

	"overgo/internal/gguf"
)

func TestLayerCatalogPlanCompilesRotaryLookupAndSelection(t *testing.T) {
	spec := Spec{
		CommonSpec: CommonSpec{ContextLength: 8192},
		AttentionSpec: AttentionSpec{
			KeyLength: 64, RopeDimensionCount: 32, RopeScalingType: ropeScalingLongRoPE,
			OriginalContextLength: 4096,
		},
	}
	profile := ArchitectureProfile{Capabilities: ArchitectureLongRoPE}
	plan := spec.rotaryCatalogPlan(profile, LayerPlan{Layer: 2})
	if plan.factor != "blk.2.rope_freqs.weight" ||
		plan.long != "blk.2.rope_factors_long.weight" || plan.longFallback != "blk.0.rope_factors_long.weight" ||
		plan.short != "blk.2.rope_factors_short.weight" || plan.shortFallback != "blk.0.rope_factors_short.weight" ||
		plan.factorElements != 32 || plan.longElements != 16 || !plan.require || !plan.selectLong {
		t.Fatalf("rotary catalog plan = %+v", plan)
	}
	catalog, err := newWeightCatalog(&gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("blk.2.rope_freqs.weight", 16),
		tensorInfo("blk.0.rope_factors_long.weight", 16),
		tensorInfo("blk.0.rope_factors_short.weight", 16),
	}})
	if err != nil {
		t.Fatal(err)
	}
	var layer LayerWeights
	if err := plan.bind(catalog, "fixture", &layer); err != nil {
		t.Fatal(err)
	}
	if layer.RopeFactors == nil || layer.RopeFactors.Name != "blk.0.rope_factors_long.weight" {
		t.Fatalf("selected factors = %+v", layer.RopeFactors)
	}
}

func TestLayerCatalogPlanCompilesSharedFactorFallback(t *testing.T) {
	plan := (Spec{AttentionSpec: AttentionSpec{KeyLength: 32}}).rotaryCatalogPlan(
		ArchitectureProfile{LayerTopology: LayerTopologySharedKVAdapter},
		LayerPlan{Layer: 3},
	)
	if plan.factor != "blk.3.rope_freqs.weight" || plan.factorFallback != "rope_freqs.weight" {
		t.Fatalf("factor lookup = %+v", plan)
	}
}
