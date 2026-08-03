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

func TestCompileModelPlanSelectsCachedGraphPolicy(t *testing.T) {
	tests := []struct {
		name    string
		spec    Spec
		weights Weights
		want    CachedGraphPolicy
	}{
		{
			name: "dense attention",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}},
			want: CachedGraphDense,
		},
		{
			name: "dense MoE",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "qwen3moe", BlockCount: 1}},
			want: CachedGraphDense,
		},
		{
			name: "sentinel cache",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "deci", BlockCount: 1}},
			want: CachedGraphDense,
		},
		{
			name: "shared KV",
			spec: Spec{
				CommonSpec:     CommonSpec{Architecture: "gemma4", BlockCount: 2},
				MultimodalSpec: MultimodalSpec{SharedKVLayers: 1},
			},
			want: CachedGraphDense,
		},
		{
			name: "AltUp",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "gemma3n", BlockCount: 1}},
			want: CachedGraphLayered,
		},
		{
			name: "MLA",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "deepseek2", BlockCount: 1}},
			want: CachedGraphLayered,
		},
		{
			name: "DSA",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "deepseek32", BlockCount: 1}},
			want: CachedGraphLayered,
		},
		{
			name: "recurrent",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "rwkv7", BlockCount: 1}},
			want: CachedGraphLayered,
		},
		{
			name: "hybrid attention-only layer",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "jamba", BlockCount: 1}},
			want: CachedGraphLayered,
		},
		{
			name:    "hybrid recurrent layer",
			spec:    Spec{CommonSpec: CommonSpec{Architecture: "jamba", BlockCount: 1}},
			weights: Weights{Layers: []LayerWeights{{Recurrent: true}}},
			want:    CachedGraphLayered,
		},
		{
			name: "encoder",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "bert", BlockCount: 1}},
			want: CachedGraphLayered,
		},
		{
			name: "empty",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "llama"}},
			want: CachedGraphLayered,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan, err := CompileModelPlan(test.spec, test.weights)
			if err != nil {
				t.Fatal(err)
			}
			if plan.CachedGraph != test.want {
				t.Fatalf("cached graph = %v, want %v; plan = %+v", plan.CachedGraph, test.want, plan)
			}
		})
	}
}

func TestCompileModelPlanAppliesSpecForwardOverride(t *testing.T) {
	plan, err := CompileModelPlan(Spec{
		CommonSpec:    CommonSpec{Architecture: "llama", BlockCount: 1},
		AttentionSpec: AttentionSpec{NonCausalAttention: true},
	}, Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Profile.Forward != ForwardNonCausal {
		t.Fatalf("forward policy = %v", plan.Profile.Forward)
	}
}

func TestCachedDenseGraphPolicyRequiresCompatibleLayers(t *testing.T) {
	for _, architecture := range SupportedArchitectures() {
		spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 1}}
		plan, err := CompileModelPlan(spec, Weights{})
		if err != nil {
			t.Fatalf("%s: %v", architecture, err)
		}
		if plan.CachedGraph != CachedGraphDense {
			continue
		}
		if plan.Profile.GraphFamily != ArchitectureFamilyAttention &&
			plan.Profile.GraphFamily != ArchitectureFamilyMoE {
			t.Fatalf("%s selected dense graph for family %v", architecture, plan.Profile.GraphFamily)
		}
		if plan.Profile.Has(ArchitectureAltUp) {
			t.Fatalf("%s selected dense graph with AltUp", architecture)
		}
		for _, layer := range plan.Layers {
			if layer.Block != BlockDense || layer.Attention != AttentionStandard ||
				layer.Cache != CacheAttention && layer.Cache != CacheSentinel {
				t.Fatalf("%s selected dense graph for layer %+v", architecture, layer)
			}
		}
	}
}

func TestCompileModelPlanBoundsAndArchitecture(t *testing.T) {
	plan, err := CompileModelPlan(Spec{
		CommonSpec:  CommonSpec{Architecture: "t5", BlockCount: 3},
		EncoderSpec: EncoderSpec{DecoderBlockCount: 2},
	}, Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Layers) != 3 || plan.CacheLayers != 2 || !plan.HasCache(CacheT5) {
		t.Fatalf("plan = %+v", plan)
	}
	if _, err := plan.Layer(3); err == nil {
		t.Fatal("out-of-range layer accepted")
	}
	_, err = CompileModelPlan(Spec{CommonSpec: CommonSpec{Architecture: "missing"}}, Weights{})
	var unsupported *UnsupportedArchitectureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %v", err)
	}
}

func TestPlanLayerPinsSharedKVSource(t *testing.T) {
	spec := Spec{
		CommonSpec: CommonSpec{Architecture: "gemma4", BlockCount: 4},
		MultimodalSpec: MultimodalSpec{
			SharedKVLayers: 2,
		},
	}
	owned := spec.PlanLayer(1, false)
	shared := spec.PlanLayer(2, false)
	if !owned.HasKV || owned.SharedKV || shared.HasKV || !shared.SharedKV || shared.KVSource != 1 {
		t.Fatalf("owned = %+v, shared = %+v", owned, shared)
	}
}
