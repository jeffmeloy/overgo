package model

import (
	"errors"
	"testing"
)

func TestPlanLayerDerivesExecutionPolicy(t *testing.T) {
	tests := []struct {
		name      string
		spec      Spec
		recurrent bool
		attention AttentionPolicy
		extent    CacheExtent
	}{
		{
			name:      "llama",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}},
			attention: AttentionStandard, extent: CacheExtentToken,
		},
		{
			name:      "dsa",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "deepseek32", BlockCount: 1}},
			attention: AttentionDSA, extent: CacheExtentToken,
		},
		{
			name: "qwen-gdn",
			spec: Spec{
				CommonSpec:    CommonSpec{Architecture: "qwen35", BlockCount: 1},
				RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true}},
			},
			attention: AttentionQwenGDN, extent: CacheExtentFixed,
		},
		{
			name:      "lfm2",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "lfm2", BlockCount: 1}},
			recurrent: true, attention: AttentionLFM2, extent: CacheExtentFixed,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan := test.spec.PlanLayer(0, test.recurrent)
			if plan.Attention != test.attention || plan.CacheExtent != test.extent {
				t.Fatalf("plan = %+v", plan)
			}
		})
	}
}

func TestCompileModelPlanPinsLayerPolicies(t *testing.T) {
	tests := []struct {
		name  string
		spec  Spec
		layer LayerWeights
		block BlockPolicy
		cache CachePolicy
	}{
		{
			name: "mamba", spec: Spec{CommonSpec: CommonSpec{Architecture: "mamba", BlockCount: 1}},
			block: BlockMamba, cache: CacheMamba,
		},
		{
			name: "jamba attention", spec: Spec{CommonSpec: CommonSpec{Architecture: "jamba", BlockCount: 1}},
			block: BlockDense, cache: CacheAttention,
		},
		{
			name: "jamba recurrent", spec: Spec{CommonSpec: CommonSpec{Architecture: "jamba", BlockCount: 1}},
			layer: LayerWeights{Recurrent: true}, block: BlockJamba, cache: CacheMamba,
		},
		{
			name: "DSA", spec: Spec{CommonSpec: CommonSpec{Architecture: "deepseek32", BlockCount: 1}},
			block: BlockDSA, cache: CacheAttention,
		},
		{
			name: "DeepSeek 4", spec: Spec{CommonSpec: CommonSpec{Architecture: "deepseek4", BlockCount: 1}},
			block: BlockDeepSeek4, cache: CacheDeepSeek4,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan, err := CompileModelPlan(test.spec, Weights{Layers: []LayerWeights{test.layer}})
			if err != nil {
				t.Fatal(err)
			}
			if len(plan.Layers) != 1 || plan.Layers[0].Block != test.block || plan.Layers[0].Cache != test.cache {
				t.Fatalf("plan = %+v", plan)
			}
		})
	}
}

func TestCompileModelPlanBoundsAndArchitecture(t *testing.T) {
	plan, err := CompileModelPlan(Spec{
		CommonSpec:  CommonSpec{Architecture: "t5", BlockCount: 1},
		EncoderSpec: EncoderSpec{DecoderBlockCount: 2},
	}, Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Layers) != 2 {
		t.Fatalf("layer count = %d", len(plan.Layers))
	}
	if _, err := plan.Layer(2); err == nil {
		t.Fatal("out-of-range layer accepted")
	}
	_, err = CompileModelPlan(Spec{CommonSpec: CommonSpec{Architecture: "missing"}}, Weights{})
	var unsupported *UnsupportedArchitectureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %v", err)
	}
}
