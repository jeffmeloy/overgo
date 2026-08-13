package model

import (
	"errors"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

func TestValidateModelPlanRejectsMutatedProgram(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}}
	plan, err := CompileModelPlan(spec, Weights{})
	if err != nil {
		t.Fatal(err)
	}
	plan.layers[0].Program.Instructions[0].Operator = LayerOperatorNone
	if err := validateModelPlan(spec, Weights{}, plan); err == nil ||
		!strings.Contains(err.Error(), "operator program is inconsistent") {
		t.Fatalf("mutated program error = %v", err)
	}
}

func TestLayerProgramOverflowCannotMasqueradeAsEmpty(t *testing.T) {
	stages := make([]LayerOperatorInstruction, maxLayerInstructions+1)
	for index := range stages {
		stages[index] = layerStage(LayerOperatorResidual)
	}
	program := newLayerProgram(stages...)
	if program.Count != maxLayerInstructions+1 || program.valid() {
		t.Fatalf("overflow program = %+v", program)
	}

	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}}
	plan, err := CompileModelPlan(spec, Weights{})
	if err != nil {
		t.Fatal(err)
	}
	plan.layers[0].Program = program
	if err := validateModelPlan(spec, Weights{}, plan); err == nil ||
		!strings.Contains(err.Error(), "instructions; capacity") {
		t.Fatalf("overflow program error = %v", err)
	}
}

func TestCompileModelPlanOwnsTerminalPolicy(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}}
	tied, err := CompileModelPlan(spec, Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if tied.Terminal().Normalization != OutputNormModel ||
		tied.Terminal().OutputHead != OutputHeadTokenEmbedding {
		t.Fatalf("tied terminal = %+v", tied.Terminal())
	}
	output := gguf.TensorInfo{Name: "output.weight"}
	dedicated, err := CompileModelPlan(spec, Weights{Output: &output})
	if err != nil {
		t.Fatal(err)
	}
	if dedicated.Terminal().OutputHead != OutputHeadDedicated {
		t.Fatalf("dedicated terminal = %+v", dedicated.Terminal())
	}
}

func TestCompileModelPlanOwnsDraftPolicy(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{
		Architecture: "step35", BlockCount: 1, NextNPredictLayers: 2,
	}}
	plan, err := CompileModelPlan(spec, Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Draft().Kind != DraftStep35MTP || plan.Draft().Heads != spec.NextNPredictLayers ||
		plan.Draft().Session != DraftSessionMulti {
		t.Fatalf("draft plan = %+v", plan.Draft())
	}
	first, firstErr := plan.DraftLayer(0)
	second, secondErr := plan.DraftLayer(1)
	if firstErr != nil || secondErr != nil || first.Layer != spec.BlockCount || second.Layer != spec.BlockCount+1 {
		t.Fatalf("draft layers = %+v/%+v (%v/%v)", first, second, firstErr, secondErr)
	}
}

func TestPlanLayerDerivesExecutionPolicy(t *testing.T) {
	tests := []struct {
		name      string
		spec      Spec
		recurrent bool
		attention AttentionPolicy
		mode      CacheStateMode
	}{
		{
			name:      "llama",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}},
			attention: AttentionStandard, mode: CacheStateToken,
		},
		{
			name:      "dsa",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "deepseek32", BlockCount: 1}},
			attention: AttentionSparseLatent, mode: CacheStateToken,
		},
		{
			name: "qwen-gdn",
			spec: Spec{
				CommonSpec:    CommonSpec{Architecture: "qwen35", BlockCount: 1},
				RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true}},
			},
			attention: AttentionGatedDelta, mode: CacheStateFixed,
		},
		{
			name:      "lfm2",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "lfm2", BlockCount: 1}},
			recurrent: true, attention: AttentionLFM2, mode: CacheStateFixed,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan := test.spec.PlanLayer(0, test.recurrent)
			if plan.Attention != test.attention || plan.CacheMode != test.mode {
				t.Fatalf("plan = %+v", plan)
			}
		})
	}
}

func TestPlanLayerCompilesTensorGraphControls(t *testing.T) {
	step := Spec{
		CommonSpec: CommonSpec{Architecture: "step35", BlockCount: 1},
		AttentionSpec: AttentionSpec{
			KeyLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10_000,
		},
		MoESpec: MoESpec{
			ExpertUsedCount: 2, ExpertWeightsScale: 1, ExpertGatingFunc: expertGatingSigmoid,
			LayerSwiGLUClamp: []float32{7},
		},
	}
	plan := step.PlanLayer(0, false)
	if plan.Rotary.kind != rotaryGraphSingle ||
		plan.Rotary.layout != tensor.RoPELayoutNeoX || plan.Rotary.factorPairs != 1 ||
		plan.Experts.Routing != tensor.MoERoutingSigmoid ||
		!plan.Experts.SelectionBias || plan.Experts.SwiGLUClamp != 7 {
		t.Fatalf("Step3.5 graph plan = %+v", plan)
	}

	gptOSS := Spec{
		CommonSpec: CommonSpec{Architecture: "gpt-oss", BlockCount: 1},
		AttentionSpec: AttentionSpec{
			SlidingWindow: 128, SlidingLayers: []bool{true},
		},
		MoESpec: MoESpec{ExpertUsedCount: 2, ExpertWeightsScale: 1},
	}
	plan = gptOSS.PlanLayer(0, false)
	if !plan.AttentionGraph.UseSinks || plan.AttentionGraph.Window != 128 ||
		plan.Experts.Routing != tensor.MoERoutingSelectedSoftmax ||
		plan.Experts.Activation != tensor.MoEActivationSwiGLUOAI ||
		plan.Experts.NormalizeTopKProb {
		t.Fatalf("GPT-OSS graph plan = %+v", plan)
	}
}

func TestCompileModelPlanPinsLayerPolicies(t *testing.T) {
	tests := []struct {
		name      string
		spec      Spec
		layer     LayerWeights
		mixer     recurrentMixerPolicy
		attention AttentionPolicy
		cache     CachePolicy
	}{
		{
			name: "mamba", spec: Spec{CommonSpec: CommonSpec{Architecture: "mamba", BlockCount: 1}},
			mixer: recurrentMixerSelectiveScan, cache: CacheMamba,
		},
		{
			name: "jamba attention", spec: Spec{CommonSpec: CommonSpec{Architecture: "jamba", BlockCount: 1}},
			cache: CacheAttention,
		},
		{
			name: "jamba recurrent", spec: Spec{CommonSpec: CommonSpec{Architecture: "jamba", BlockCount: 1}},
			layer: LayerWeights{Recurrent: true}, mixer: recurrentMixerWeightedSelectiveScan, cache: CacheMamba,
		},
		{
			name: "DSA", spec: Spec{CommonSpec: CommonSpec{Architecture: "deepseek32", BlockCount: 1}},
			attention: AttentionSparseLatent, cache: CacheAttention,
		},
		{
			name: "DeepSeek 4", spec: Spec{
				CommonSpec:    CommonSpec{Architecture: "deepseek4", BlockCount: 1},
				AttentionSpec: AttentionSpec{CompressRatios: []uint32{0}},
			},
			cache: CacheDeepSeek4,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan, err := CompileModelPlan(test.spec, Weights{Layers: []LayerWeights{test.layer}})
			if err != nil {
				t.Fatal(err)
			}
			if plan.LayerCount() != 1 {
				t.Fatalf("plan = %+v", plan)
			}
			layer := plan.layers[0]
			if layer.Mixer != test.mixer || layer.Attention != test.attention ||
				layer.Cache != test.cache {
				t.Fatalf("plan = %+v", plan)
			}
		})
	}
}

func TestCompileModelPlanWithProfilePinsResolvedPolicy(t *testing.T) {
	profile, ok := LookupArchitecture("llama")
	if !ok {
		t.Fatal("llama profile is absent")
	}
	profile.Attention = AttentionLFM2
	plan, err := CompileModelPlanWithProfile(
		Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}}, Weights{}, profile,
	)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Profile().Attention != AttentionLFM2 || plan.layers[0].Attention != AttentionLFM2 {
		t.Fatalf("resolved profile was not pinned: %+v", plan)
	}

	profile.Name = "qwen"
	if _, err := CompileModelPlanWithProfile(
		Spec{CommonSpec: CommonSpec{Architecture: "llama"}}, Weights{}, profile,
	); err == nil {
		t.Fatal("mismatched profile accepted")
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
			if plan.CachedGraph() != test.want {
				t.Fatalf("cached graph = %v, want %v; plan = %+v", plan.CachedGraph(), test.want, plan)
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
	if plan.Forward().Operation != ForwardOperationBidirectional {
		t.Fatalf("forward program = %v", plan.Forward())
	}
}

func TestCachedLayerTopologyRequiresCompatibleLayers(t *testing.T) {
	for _, architecture := range SupportedArchitectures() {
		spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 1}}
		if architecture == "deepseek4" {
			spec.CompressRatios = []uint32{0}
		}
		plan, err := CompileModelPlan(spec, Weights{})
		if err != nil {
			t.Fatalf("%s: %v", architecture, err)
		}
		if plan.CachedGraph() != CachedGraphDense {
			continue
		}
		if plan.Profile().GraphFamily != ArchitectureFamilyAttention &&
			plan.Profile().GraphFamily != ArchitectureFamilyMoE {
			t.Fatalf("%s selected dense graph for family %v", architecture, plan.Profile().GraphFamily)
		}
		if plan.Profile().Has(ArchitectureAltUp) {
			t.Fatalf("%s selected dense graph with AltUp", architecture)
		}
		for _, layer := range plan.layers {
			if layer.Mixer != recurrentMixerNone || layer.Attention != AttentionStandard ||
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
	if plan.LayerCount() != 3 || plan.CacheLayerCount() != 2 || !plan.HasCache(CacheCrossAttention) {
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

func TestCompileModelPlanRejectsCrossPolicyConflicts(t *testing.T) {
	tests := []struct {
		name string
		spec Spec
	}{
		{
			name: "shared KV extent",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "gemma4", BlockCount: 2},
				MultimodalSpec: MultimodalSpec{SharedKVLayers: 2}},
		},
		{
			name: "deepstack source",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "granite", BlockCount: 2},
				MultimodalSpec: MultimodalSpec{DeepstackLayerCount: 1, DeepstackMapping: []int32{0, 2}}},
		},
		{
			name: "auxiliary ordering",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "glm-dsa", BlockCount: 2},
				AttentionSpec: AttentionSpec{IndexerFullLayers: []bool{false, true}}},
		},
		{
			name: "DeepSeek4 cache schema",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "deepseek4", BlockCount: 1}},
		},
		{
			name: "temperature",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "mistral3", BlockCount: 1},
				AttentionSpec: AttentionSpec{AttentionTempScale: 0.1}},
		},
		{
			name: "draft family",
			spec: Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1, NextNPredictLayers: 1}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := CompileModelPlan(test.spec, Weights{}); err == nil {
				t.Fatalf("CompileModelPlan(%+v) succeeded", test.spec)
			}
		})
	}
	valid := Spec{
		CommonSpec:    CommonSpec{Architecture: "glm-dsa", BlockCount: 2},
		AttentionSpec: AttentionSpec{IndexerFullLayers: []bool{true, false}},
	}
	if _, err := CompileModelPlan(valid, Weights{}); err != nil {
		t.Fatalf("valid auxiliary flow: %v", err)
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

func TestPlanLayerCompilesProjectedStreams(t *testing.T) {
	granite := Spec{
		CommonSpec: CommonSpec{Architecture: "granite", BlockCount: 4},
		MultimodalSpec: MultimodalSpec{
			DeepstackLayerCount: 2, DeepstackMapping: []int32{0, 2, -1, 1},
		},
	}
	want := []DeepstackSource{DeepstackSourceNone, 1, DeepstackSourceNone, 0}
	for layer, source := range want {
		plan := granite.PlanLayer(uint32(layer), false)
		if plan.DeepstackBefore != source || plan.DeepstackAfter != DeepstackSourceNone {
			t.Fatalf("Granite layer %d deepstack = %d/%d, want %d/none", layer, plan.DeepstackBefore, plan.DeepstackAfter, source)
		}
	}
	qwen := Spec{
		CommonSpec:     CommonSpec{Architecture: "qwen3vl", BlockCount: 3},
		MultimodalSpec: MultimodalSpec{DeepstackLayerCount: 2},
	}
	for layer := range uint32(3) {
		plan := qwen.PlanLayer(layer, false)
		wantAfter := DeepstackSourceNone
		if layer < 2 {
			wantAfter = DeepstackSource(layer)
		}
		if plan.DeepstackBefore != DeepstackSourceNone || plan.DeepstackAfter != wantAfter {
			t.Fatalf("Qwen3-VL layer %d deepstack = %d/%d, want none/%d", layer, plan.DeepstackBefore, plan.DeepstackAfter, wantAfter)
		}
	}
}

func TestPlanLayerCompilesAuxiliaryFlow(t *testing.T) {
	rwkv := Spec{CommonSpec: CommonSpec{Architecture: "rwkv7", BlockCount: 2}}
	first, second := rwkv.PlanLayer(0, false), rwkv.PlanLayer(1, false)
	if first.AuxiliaryInput != AuxiliaryNone || first.AuxiliaryOutput != AuxiliaryRWKVValue ||
		second.AuxiliaryInput != AuxiliaryRWKVValue || second.AuxiliaryOutput != AuxiliaryNone {
		t.Fatalf("RWKV auxiliary plans = %+v / %+v", first, second)
	}
	dsa := Spec{
		CommonSpec:    CommonSpec{Architecture: "glm-dsa", BlockCount: 3},
		AttentionSpec: AttentionSpec{IndexerFullLayers: []bool{true, false, true}},
	}
	for layer, wantInput := range []AuxiliaryFlow{AuxiliaryNone, AuxiliarySparseTopK, AuxiliaryNone} {
		plan := dsa.PlanLayer(uint32(layer), false)
		if plan.AuxiliaryInput != wantInput || plan.AuxiliaryOutput != AuxiliarySparseTopK {
			t.Fatalf("GLM-DSA layer %d auxiliary = %v/%v", layer, plan.AuxiliaryInput, plan.AuxiliaryOutput)
		}
	}
}

func TestPlanLayerCompilesAttentionTemperature(t *testing.T) {
	configured := Spec{CommonSpec: CommonSpec{Architecture: "mistral3", BlockCount: 1}}
	if got := configured.PlanLayer(0, false).Temperature; got != AttentionTemperatureConfigured {
		t.Fatalf("Mistral3 temperature = %v", got)
	}
	llama4 := Spec{
		CommonSpec:    CommonSpec{Architecture: "llama4", BlockCount: 2},
		AttentionSpec: AttentionSpec{NoRopeLayerStep: 2},
	}
	if got := llama4.PlanLayer(0, false).Temperature; got != AttentionTemperatureNone {
		t.Fatalf("Llama4 RoPE layer temperature = %v", got)
	}
	if got := llama4.PlanLayer(1, false).Temperature; got != AttentionTemperatureNoRoPE {
		t.Fatalf("Llama4 no-RoPE layer temperature = %v", got)
	}
}

func TestPlanLayerCompilesSideInputPolicies(t *testing.T) {
	gemma4 := Spec{
		CommonSpec:     CommonSpec{Architecture: "gemma4"},
		MultimodalSpec: MultimodalSpec{EmbeddingPerLayer: 2},
	}
	plan := gemma4.PlanLayer(0, false)
	if !plan.PerLayerInput || plan.AttentionBlocks != AttentionBlocksUncached || plan.EmbeddingSkip {
		t.Fatalf("Gemma4 side-input plan = %+v", plan)
	}
	talkie := Spec{CommonSpec: CommonSpec{Architecture: "talkie"}}.PlanLayer(0, false)
	if !talkie.EmbeddingSkip || talkie.PerLayerInput || talkie.AttentionBlocks != AttentionBlocksNone {
		t.Fatalf("Talkie side-input plan = %+v", talkie)
	}
}

func TestNormPlanCompilesOperationPlacementBiasAndLayout(t *testing.T) {
	tests := []struct {
		architecture string
		epsilon      float32
		operation    NormalizationPolicy
		pre          bool
		post         bool
		bias         bool
		postLayout   PostNormLayoutPolicy
		ffnLayout    FeedForwardNormLayoutPolicy
	}{
		{"bert", 1e-5, NormalizationLayer, false, true, true, PostNormLayoutBERT, FeedForwardNormLayoutStandard},
		{"olmo2", 0, NormalizationRMS, false, true, false, PostNormLayoutStandard, FeedForwardNormLayoutStandard},
		{"grok", 0, NormalizationRMS, true, true, false, PostNormLayoutGrok, FeedForwardNormLayoutStandard},
		{"dbrx", 1e-5, NormalizationLayer, true, false, false, PostNormLayoutStandard, FeedForwardNormLayoutAttentionOutput},
		{"falcon-h1", 0, NormalizationRMS, true, false, false, PostNormLayoutStandard, FeedForwardNormLayoutBare},
		{"talkie", 0, NormalizationUnweightedRMS, true, false, false, PostNormLayoutStandard, FeedForwardNormLayoutStandard},
	}
	for _, test := range tests {
		spec := Spec{CommonSpec: CommonSpec{Architecture: test.architecture, LayerNormEpsilon: test.epsilon}}
		plan := spec.NormPlan()
		if plan.Operation != test.operation || plan.PreAttention != test.pre ||
			plan.PreFeedForward != test.pre || plan.PostAttention != test.post ||
			plan.PostFeedForward != test.post || plan.Bias != test.bias ||
			plan.PostNormLayout != test.postLayout || plan.FeedForwardLayout != test.ffnLayout {
			t.Errorf("%s normalization plan = %+v", test.architecture, plan)
		}
	}
}
