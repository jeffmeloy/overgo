package model

import (
	"llamacpp2go/internal/tensor"

	"llamacpp2go/internal/tensor/dtype"

	"slices"

	"testing"
)

func TestBuildNormalRoPELongRoPEUsesFactorsAndAttentionScale(t *testing.T) {
	for _, architecture := range []string{"llama", "llama-embed", "minicpm", "mistral3"} {
		t.Run(architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{
				Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 12,
				HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeDimensionCount: 4, RopeFrequencyBase: 10000,
				RopeScalingType: "longrope", RopeAttentionFactor: 1.25,
				RMSNormEpsilon: 1e-6,
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := denseBlockInputs(builder, spec)
			weights.AttentionQNorm = nil
			weights.AttentionKNorm = nil
			weights.RopeFactors = builder.Input("rope_long", dtype.F32, tensor.MustShape(2))
			output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(output)
			if err != nil {
				t.Fatal(err)
			}
			var ropeCount, attentionScales int
			for _, node := range nodes {
				if node.Op == tensor.OpRoPENormal {
					ropeCount++
					if len(node.Inputs) != 2 || node.Inputs[1] != weights.RopeFactors {
						t.Fatal("LongRoPE factors are disconnected")
					}
				}
				if node.Op == tensor.OpScale && node.Attrs.(tensor.ScaleAttributes).Value == 1.25 {
					attentionScales++
				}
			}
			if ropeCount != 2 || attentionScales != 2 {
				t.Fatalf("%s LongRoPE graph has rope=%d scale=%d", architecture, ropeCount, attentionScales)
			}
		})
	}
}

func TestBuildNormalRoPEYaRNUsesFactors(t *testing.T) {
	for _, architecture := range []string{"llama", "llama-embed", "minicpm", "mistral3"} {
		t.Run(architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{
				Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 12,
				HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeDimensionCount: 4, RopeFrequencyBase: 10000,
				RopeScalingType: "yarn", RopeScalingFactor: 4, OriginalContextLength: 16,
				YaRNExtFactor: 1, YaRNAttentionFactor: 0.9, YaRNBetaFast: 16, YaRNBetaSlow: 2,
				RMSNormEpsilon: 1e-6,
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := denseBlockInputs(builder, spec)
			weights.AttentionQNorm = nil
			weights.AttentionKNorm = nil
			weights.RopeFactors = builder.Input("rope_factors", dtype.F32, tensor.MustShape(2))
			output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 17})
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(output)
			if err != nil {
				t.Fatal(err)
			}
			var ropeCount int
			for _, node := range nodes {
				if node.Op != tensor.OpRoPENormal {
					continue
				}
				ropeCount++
				attributes := node.Attrs.(tensor.RoPEAttributes)
				if len(node.Inputs) != 2 || node.Inputs[1] != weights.RopeFactors ||
					attributes.OriginalContext != 16 || attributes.FrequencyScale != 0.25 ||
					attributes.ExtFactor != 1 || attributes.AttentionFactor != 0.9 ||
					attributes.BetaFast != 16 || attributes.BetaSlow != 2 {
					t.Fatalf("unexpected %s YaRN node: inputs=%v attrs=%+v", architecture, node.Inputs, attributes)
				}
			}
			if ropeCount != 2 {
				t.Fatalf("%s YaRN RoPE count = %d, want 2", architecture, ropeCount)
			}
		})
	}
}

func TestBuildDenseOLMo2PostNormalizedSlidingBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "olmo2",
		BlockCount:        4,
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeFrequencySWA:  10000,
		RopeScalingType:   "linear",
		RopeScalingFactor: 4,
		RMSNormEpsilon:    1e-6,
		SlidingWindow:     1024,
		SlidingPattern:    4,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNorm = nil
	weights.FeedForwardNorm = nil
	weights.AttentionQNorm = builder.Input(
		"attn_q_norm",
		dtype.F32,
		tensor.MustShape(8),
	)
	weights.AttentionKNorm = builder.Input(
		"attn_k_norm",
		dtype.F32,
		tensor.MustShape(4),
	)
	weights.AttentionPostNorm = builder.Input(
		"post_attention_norm",
		dtype.F32,
		tensor.MustShape(8),
	)
	weights.FeedForwardPostNorm = builder.Input(
		"post_ffw_norm",
		dtype.F32,
		tensor.MustShape(8),
	)
	result, err := BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	var attentionCount, ropeCount int
	for _, node := range nodes {
		found[node.Name] = true
		switch node.Op {
		case tensor.OpAttention:
			attentionCount++
			attributes := node.Attrs.(tensor.AttentionAttributes)
			if attributes.Window != 1024 {
				t.Fatalf("OLMo2 attention window = %d, want 1024", attributes.Window)
			}
		case tensor.OpRoPENeoX:
			ropeCount++
			attributes := node.Attrs.(tensor.RoPEAttributes)
			if attributes.FrequencyScale != 1 {
				t.Fatalf("OLMo2 sliding RoPE scale = %v, want 1", attributes.FrequencyScale)
			}
		}
	}
	for _, name := range []string{
		"attn_q_norm",
		"attn_k_norm",
		"post_attention_norm",
		"post_ffw_norm",
	} {
		if !found[name] {
			t.Fatalf("OLMo2 norm %q is disconnected", name)
		}
	}
	if attentionCount != 1 || ropeCount != 2 {
		t.Fatalf("OLMo2 graph has attention=%d RoPE=%d, want 1/2", attentionCount, ropeCount)
	}
}

func TestBuildDenseSmolLM3SkipsPeriodicRoPE(t *testing.T) {
	for _, test := range []struct {
		layer    uint32
		wantRoPE int
	}{
		{2, 2},
		{3, 0},
	} {
		builder := tensor.NewBuilder()
		spec := Spec{
			Architecture:      "smollm3",
			BlockCount:        36,
			EmbeddingLength:   8,
			FeedForwardLength: 12,
			HeadCount:         2,
			HeadCountKV:       1,
			KeyLength:         4,
			ValueLength:       4,
			RopeFrequencyBase: 10000,
			AttentionScale:    0.25,
			RMSNormEpsilon:    1e-6,
			NoRopeLayerStep:   4,
		}
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := denseBlockInputs(builder, spec)
		weights.AttentionQNorm = nil
		weights.AttentionKNorm = nil
		result, err := BuildDenseBlockCachedForLayer(
			builder,
			input,
			spec,
			weights,
			[]uint32{0, 1},
			nil,
			nil,
			test.layer,
		)
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := tensor.Topological(result.Output)
		if err != nil {
			t.Fatal(err)
		}
		var ropeCount int
		for _, node := range nodes {
			if node.Op == tensor.OpRoPENormal {
				ropeCount++
			}
			if node.Op == tensor.OpAttention {
				attributes := node.Attrs.(tensor.AttentionAttributes)
				if attributes.Scale != 0.25 {
					t.Fatalf("SmolLM3 attention scale = %v, want 0.25", attributes.Scale)
				}
			}
		}
		if ropeCount != test.wantRoPE {
			t.Fatalf("SmolLM3 layer %d RoPE count = %d, want %d", test.layer, ropeCount, test.wantRoPE)
		}
	}
}

func TestBuildDenseMiniCPMScalesResidualBranches(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "minicpm",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		ResidualScale:     0.25,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var residualScales int
	for _, node := range nodes {
		if node.Op == tensor.OpScale &&
			node.Attrs.(tensor.ScaleAttributes).Value == 0.25 {
			residualScales++
		}
	}
	if residualScales != 2 {
		t.Fatalf("MiniCPM residual scale count = %d, want 2", residualScales)
	}
}

func TestBuildDenseGraniteHonorsRoPESwitchAndScales(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "granite",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		AttentionScale:    0.25,
		ResidualScale:     0.5,
		RMSNormEpsilon:    1e-6,
		RopeDisabled:      true,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var ropeCount, residualScales int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENormal {
			ropeCount++
		}
		if node.Op == tensor.OpScale &&
			node.Attrs.(tensor.ScaleAttributes).Value == 0.5 {
			residualScales++
		}
		if node.Op == tensor.OpAttention &&
			node.Attrs.(tensor.AttentionAttributes).Scale != 0.25 {
			t.Fatalf("Granite attention scale was not honored")
		}
	}
	if ropeCount != 0 || residualScales != 2 {
		t.Fatalf("Granite graph has RoPE=%d residual scales=%d, want 0/2", ropeCount, residualScales)
	}
}

func TestBuildDenseMaincoderNormalizesQKAfterRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "maincoder",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range nodes {
		if node.Op != tensor.OpAttention {
			continue
		}
		for index, qk := range node.Inputs[:2] {
			if qk.Op != tensor.OpMultiply ||
				len(qk.Inputs) != 2 ||
				qk.Inputs[0].Op != tensor.OpRMSNorm ||
				qk.Inputs[0].Inputs[0].Op != tensor.OpRoPENormal {
				t.Fatalf("Maincoder attention input %d is not post-RoPE normalized", index)
			}
		}
		return
	}
	t.Fatal("Maincoder attention node was not found")
}

func TestBuildDenseOrionUsesAffineLayerNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "orion",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNormCount, ropeCount int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNormCount++
		case tensor.OpRoPENeoX:
			ropeCount++
		}
	}
	if layerNormCount != 2 || ropeCount != 2 {
		t.Fatalf(
			"Orion graph has LayerNorm=%d RoPE=%d, want 2/2",
			layerNormCount,
			ropeCount,
		)
	}
}

func TestBuildDenseStarCoder2UsesSequentialGELU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "starcoder2",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionOutputBias = builder.Input(
		"attn_output_bias", dtype.F32, tensor.MustShape(8),
	)
	weights.FeedForwardUpBias = builder.Input(
		"ffn_up_bias", dtype.F32, tensor.MustShape(12),
	)
	weights.FeedForwardDownBias = builder.Input(
		"ffn_down_bias", dtype.F32, tensor.MustShape(8),
	)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNormCount, ropeCount, geluCount, siluCount int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNormCount++
		case tensor.OpRoPENeoX:
			ropeCount++
		case tensor.OpGELU:
			geluCount++
		case tensor.OpSiLU:
			siluCount++
		}
		if node == weights.FeedForwardGate {
			t.Fatal("StarCoder2 graph unexpectedly consumes an FFN gate")
		}
	}
	if layerNormCount != 2 || ropeCount != 2 || geluCount != 1 || siluCount != 0 {
		t.Fatalf(
			"StarCoder2 graph has LayerNorm=%d RoPE=%d GELU=%d SiLU=%d, want 2/2/1/0",
			layerNormCount,
			ropeCount,
			geluCount,
			siluCount,
		)
	}
}

func TestBuildDenseCodeShellUsesSequentialGELU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "codeshell",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 1))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardUpBias = builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var geluCount int
	for _, node := range nodes {
		if node == weights.FeedForwardGate || node.Op == tensor.OpSiLU {
			t.Fatal("CodeShell graph unexpectedly consumes a gated SwiGLU path")
		}
		if node.Op == tensor.OpGELU {
			geluCount++
		}
	}
	if geluCount != 1 {
		t.Fatalf("CodeShell GELU count = %d, want 1", geluCount)
	}
}

func TestBuildDenseBaichuan7BUsesNormalRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "baichuan",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       2,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	output, err := BuildDenseBlock(
		builder,
		input,
		spec,
		denseBlockInputs(builder, spec),
		[]uint32{0, 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var normalRoPE, neoXRoPE int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpRoPENeoX:
			neoXRoPE++
		}
	}
	if normalRoPE != 2 || neoXRoPE != 0 {
		t.Fatalf("Baichuan RoPE count normal/NeoX = %d/%d, want 2/0", normalRoPE, neoXRoPE)
	}
}

func TestBuildDenseBaichuan13BUsesALiBi(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "baichuan", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RMSNormEpsilon: 1e-6, RopeDisabled: true, MaxALiBiBias: 8,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	output, err := BuildDenseBlock(
		builder, input, spec, denseBlockInputs(builder, spec), []uint32{0, 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var alibi int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENormal || node.Op == tensor.OpRoPENeoX {
			t.Fatal("Baichuan 13B graph unexpectedly contains RoPE")
		}
		if node.Op == tensor.OpAttention &&
			node.Attrs.(tensor.AttentionAttributes).MaxALiBiBias == 8 {
			alibi++
		}
	}
	if alibi != 1 {
		t.Fatalf("Baichuan 13B ALiBi attention count = %d, want 1", alibi)
	}
}

func TestBuildDenseArceeUsesSquaredReLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "arcee",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-5,
		AttentionScale:    0.25,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var squaredReLU, normalRoPE int
	for _, node := range nodes {
		if node == weights.FeedForwardGate {
			t.Fatal("Arcee graph unexpectedly consumes an FFN gate")
		}
		switch node.Op {
		case tensor.OpReLUSquared:
			squaredReLU++
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpAttention:
			if node.Attrs.(tensor.AttentionAttributes).Scale != 0.25 {
				t.Fatal("Arcee metadata attention scale was not honored")
			}
		}
	}
	if squaredReLU != 1 || normalRoPE != 2 {
		t.Fatalf("Arcee graph squared-ReLU/RoPE = %d/%d, want 1/2", squaredReLU, normalRoPE)
	}
}

func TestBuildDenseNemotronUsesAffineNormAndSquaredReLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "nemotron",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNorm, squaredReLU, neoXRoPE int
	for _, node := range nodes {
		if node == weights.FeedForwardGate {
			t.Fatal("Nemotron graph unexpectedly consumes an FFN gate")
		}
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpReLUSquared:
			squaredReLU++
		case tensor.OpRoPENeoX:
			neoXRoPE++
		}
	}
	if layerNorm != 2 || squaredReLU != 1 || neoXRoPE != 2 {
		t.Fatalf(
			"Nemotron graph LayerNorm/squared-ReLU/NeoX = %d/%d/%d, want 2/1/2",
			layerNorm,
			squaredReLU,
			neoXRoPE,
		)
	}
}

func TestBuildDenseJais2UsesBiasesAndSquaredReLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "jais2", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQBias = builder.Input("q_bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionKBias = builder.Input("k_bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionVBias = builder.Input("v_bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionOutputBias = builder.Input("o_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardUpBias = builder.Input("up_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("down_bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNorm, squaredReLU, neoX int
	for _, node := range nodes {
		if node == weights.FeedForwardGate {
			t.Fatal("Jais2 graph unexpectedly consumes an FFN gate")
		}
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpReLUSquared:
			squaredReLU++
		case tensor.OpRoPENeoX:
			neoX++
		}
	}
	if layerNorm != 2 || squaredReLU != 1 || neoX != 2 {
		t.Fatalf("Jais2 graph LayerNorm/squared-ReLU/NeoX = %d/%d/%d", layerNorm, squaredReLU, neoX)
	}
}

func TestBuildDenseJaisUsesBiasedFusedQKVAndALiBi(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "jais", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, AttentionScale: 0.25, MaxALiBiBias: 8, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQKVBias = builder.Input("qkv_bias", dtype.F32, tensor.MustShape(24))
	weights.AttentionOutputBias = builder.Input("output_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardGateBias = builder.Input("gate_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardUpBias = builder.Input("up_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("down_bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var layerNorm, rope, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpRoPENormal, tensor.OpRoPENeoX:
			rope++
		case tensor.OpSiLU:
			silu++
		}
	}
	if attention == nil {
		t.Fatal("Jais graph is missing attention")
	}
	attrs := attention.Attrs.(tensor.AttentionAttributes)
	if attrs.Scale != 0.25 || attrs.MaxALiBiBias != 8 || !attrs.Causal ||
		layerNorm != 2 || rope != 0 || silu != 1 {
		t.Fatalf("unexpected Jais graph: attention=%+v norm=%d rope=%d silu=%d", attrs, layerNorm, rope, silu)
	}
}

func TestBuildDenseOLMoUsesUnweightedLayerNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "olmo", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNorm = nil
	weights.AttentionNormBias = nil
	weights.FeedForwardNorm = nil
	weights.FeedForwardNormBias = nil
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNorm, multiplyNorm, normalRoPE int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpMultiply:
			for _, inputNode := range node.Inputs {
				if inputNode.Op == tensor.OpLayerNorm {
					multiplyNorm++
				}
			}
		}
	}
	if layerNorm != 2 || multiplyNorm != 0 || normalRoPE != 2 {
		t.Fatalf("OLMo graph LayerNorm/weighted/normal-RoPE = %d/%d/%d", layerNorm, multiplyNorm, normalRoPE)
	}
}

func TestBuildDenseSeedOSSUsesNeoXAndAttentionScale(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "seed_oss", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5, AttentionScale: 0.25,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	output, err := BuildDenseBlock(builder, input, spec, denseBlockInputs(builder, spec), []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var neoX int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENeoX {
			neoX++
		}
		if node.Op == tensor.OpAttention && node.Attrs.(tensor.AttentionAttributes).Scale != 0.25 {
			t.Fatal("Seed-OSS attention scale was not honored")
		}
	}
	if neoX != 2 {
		t.Fatalf("Seed-OSS NeoX count = %d, want 2", neoX)
	}
}

func TestBuildDenseCohere2UsesParallelResidualAndPartialSlidingRoPE(t *testing.T) {
	for _, test := range []struct {
		layer      uint32
		wantRoPE   int
		wantWindow uint32
	}{
		{layer: 0, wantRoPE: 2, wantWindow: 128},
		{layer: 3, wantRoPE: 0, wantWindow: 0},
	} {
		builder := tensor.NewBuilder()
		spec := Spec{
			Architecture: "cohere2", BlockCount: 4, EmbeddingLength: 8,
			FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
			RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
			LayerNormEpsilon: 1e-5, SlidingWindow: 128,
			SlidingPattern: 4, NoRopeLayerStep: 4,
		}
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := denseBlockInputs(builder, spec)
		weights.FeedForwardNorm = nil
		weights.FeedForwardNormBias = nil
		output, err := BuildDenseBlockCachedForLayer(
			builder, input, spec, weights, []uint32{0, 1}, nil, nil, test.layer,
		)
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := tensor.Topological(output.Output)
		if err != nil {
			t.Fatal(err)
		}
		var ropeCount, layerNormCount int
		var window uint32
		var attentionInput, feedForwardInput *tensor.Tensor
		for _, node := range nodes {
			switch node.Op {
			case tensor.OpLayerNorm:
				layerNormCount++
			case tensor.OpRoPENormal:
				ropeCount++
				attributes := node.Attrs.(tensor.RoPEAttributes)
				if attributes.RotaryDimensions != 2 || attributes.FrequencyBase != 20000 {
					t.Fatalf("unexpected Cohere2 RoPE attributes: %+v", attributes)
				}
			case tensor.OpAttention:
				window = node.Attrs.(tensor.AttentionAttributes).Window
			case tensor.OpMulMat:
				if node.Inputs[0].Name == "attn_q" {
					attentionInput = node.Inputs[1]
				}
				if node.Inputs[0].Name == "ffn_up" {
					feedForwardInput = node.Inputs[1]
				}
			}
		}
		if ropeCount != test.wantRoPE || window != test.wantWindow || layerNormCount != 1 {
			t.Fatalf(
				"Cohere2 layer %d has RoPE=%d window=%d LayerNorm=%d",
				test.layer, ropeCount, window, layerNormCount,
			)
		}
		if attentionInput == nil || attentionInput != feedForwardInput {
			t.Fatal("Cohere2 attention and FFN do not share the normalized block input")
		}
	}
}

func TestBuildCohere2MoEBlockFusedExpertsAndSharedBranch(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "cohere2moe", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		RMSNormEpsilon: 1e-5, SlidingWindow: 128,
		SlidingLayers: []bool{false, true}, LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 2, ExpertWeightsNorm: true, ExpertWeightsScale: 1.25,
		SharedExpertCount: 1, SharedExpertFF: 6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:             builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(6, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var normalized, attentionInput *tensor.Tensor
	var rope, silu, halfScale int
	var window uint32
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
			normalized = node.Inputs[0]
		case tensor.OpRoPENormal:
			rope++
			if node.Attrs.(tensor.RoPEAttributes).FrequencyBase != 20000 {
				t.Fatalf("unexpected Cohere2-MoE RoPE: %+v", node.Attrs)
			}
		case tensor.OpAttention:
			window = node.Attrs.(tensor.AttentionAttributes).Window
		case tensor.OpSiLU:
			silu++
		case tensor.OpScale:
			if node.Attrs.(tensor.ScaleAttributes).Value == 0.5 {
				halfScale++
			}
		case tensor.OpMulMat:
			if node.Inputs[0].Name == "attn_qkv" {
				attentionInput = node.Inputs[1]
			}
		}
	}
	if moe == nil || !moe.Attrs.(tensor.MoEAttributes).FusedGateUp ||
		moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		!moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb ||
		moe.Attrs.(tensor.MoEAttributes).Scale != 1.25 || rope != 2 || window != 128 ||
		silu != 1 || halfScale != 1 || normalized == nil || normalized != attentionInput {
		t.Fatalf("unexpected Cohere2-MoE graph: MoE=%+v RoPE=%d window=%d SiLU=%d half=%d", moe, rope, window, silu, halfScale)
	}
}

func TestBuildHYV3MoEBlockFusedExpertsAndPreRoPEQKNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "hy_v3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 2, ExpertWeightsNorm: true, ExpertWeightsScale: 1.25,
		SharedExpertFF: 6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:             builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionQNorm:           builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:           builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:          builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:    builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(6, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var rope, preNormRoPE, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
			if node.Inputs[0].Op == tensor.OpMultiply && node.Inputs[0].Inputs[0].Op == tensor.OpRMSNorm {
				preNormRoPE++
			}
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil || !moe.Attrs.(tensor.MoEAttributes).FusedGateUp ||
		moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		!moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb ||
		moe.Attrs.(tensor.MoEAttributes).Scale != 1.25 || len(moe.Inputs) != 6 ||
		rope != 2 || preNormRoPE != 2 || silu != 1 {
		t.Fatalf("unexpected HY-V3 graph: MoE=%+v RoPE=%d pre-norm=%d SiLU=%d", moe, rope, preNormRoPE, silu)
	}
}

func TestBuildDeepSeek2OCRMoEBlockUsesSplitQKVAndSharedExpert(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "deepseek2-ocr", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 1, ExpertWeightsScale: 1, SharedExpertFF: 12,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:               builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:               builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:               builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:          builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:    builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var rope, qkNorm, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
		case tensor.OpRMSNorm:
			if node.Name == "q_norm" || node.Name == "k_norm" {
				qkNorm++
			}
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil || !moe.Attrs.(tensor.MoEAttributes).FusedGateUp ||
		moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSoftmax ||
		moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb ||
		moe.Attrs.(tensor.MoEAttributes).Scale != 1 || len(moe.Inputs) != 6 ||
		rope != 2 || qkNorm != 0 || silu != 1 {
		t.Fatalf("unexpected DeepSeek2-OCR graph: MoE=%+v RoPE=%d Q/K norm=%d SiLU=%d", moe, rope, qkNorm, silu)
	}
}

func TestBuildErnie45MoEBlockUsesBiasedUngatedExpertsAndSharedBranch(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "ernie4_5-moe", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true,
		LeadingDenseBlocks: 1, MoELayerStep: 2, SharedExpertFF: 5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 5)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 5)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(5, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var normalRoPE, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil {
		t.Fatal("ERNIE MoE node is missing")
	}
	attributes := moe.Attrs.(tensor.MoEAttributes)
	if attributes.Gated || attributes.Routing != tensor.MoERoutingSoftmax ||
		!attributes.NormalizeTopKProb || attributes.Scale != 1.25 ||
		len(moe.Inputs) != 6 || normalRoPE != 2 || silu != 1 {
		t.Fatalf("unexpected ERNIE MoE graph: MoE=%+v RoPE=%d SiLU=%d", moe, normalRoPE, silu)
	}
}

func TestBuildPaddleOCRBlockUsesMRoPEAndOutputBias(t *testing.T) {
	testBuildMRoPETextDecoderBlock(t, "paddleocr")
}

func TestBuildQwen2VLBlockUsesMRoPEAndOutputBias(t *testing.T) {
	testBuildMRoPETextDecoderBlock(t, "qwen2vl")
}

func TestBuildQwen3VLBlockUsesQKNormAndMRoPE(t *testing.T) {
	testBuildMRoPETextDecoderBlock(t, "qwen3vl")
}

func TestBuildQwen3VLBlockUsesDistinctMRoPEPositions(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "qwen3vl", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0},
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
	positions := [4][]uint32{{10, 11}, {20, 21}, {30, 31}, {40, 41}}
	result, err := BuildDenseBlockCachedForLayerWithMultiPositions(
		builder, input, spec, weights, positions, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, node := range nodes {
		if node.Op != tensor.OpRoPEMulti {
			continue
		}
		count++
		attrs := node.Attrs.(tensor.RoPEMultiAttributes)
		for axis := range positions {
			if !slices.Equal(attrs.Positions[axis], positions[axis]) {
				t.Fatalf("MRoPE axis %d = %v, want %v", axis, attrs.Positions[axis], positions[axis])
			}
		}
	}
	if count != 2 {
		t.Fatalf("MRoPE nodes = %d, want 2", count)
	}
}

func TestBuildQwen3VLMoEBlockUsesQKNormMRoPEAndNormalizedExperts(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "qwen3vlmoe", EmbeddingLength: 8, FeedForwardLength: 24,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1.25,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0},
		RopeFrequencyBase: 10000, RopeScalingType: "linear", RopeScalingFactor: 4,
		RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
	weights.FeedForwardRouter = builder.Input("ffn_router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("ffn_gate_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardUpExperts = builder.Input("ffn_up_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardDownExperts = builder.Input("ffn_down_exps", dtype.F32, tensor.MustShape(12, 8, 4))
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var multiRoPE, rmsNorm int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPEMulti:
			multiRoPE++
		case tensor.OpRMSNorm:
			rmsNorm++
		}
	}
	if moe == nil {
		t.Fatal("Qwen3-VL-MoE graph is missing MoE")
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if !attrs.Gated || attrs.Routing != tensor.MoERoutingSoftmax || !attrs.NormalizeTopKProb ||
		attrs.TopK != 2 || attrs.Scale != 1.25 || multiRoPE != 2 || rmsNorm != 4 {
		t.Fatalf("unexpected Qwen3-VL-MoE graph: MoE=%+v MRoPE=%d RMS=%d", attrs, multiRoPE, rmsNorm)
	}
}

func testBuildMRoPETextDecoderBlock(t *testing.T, architecture string) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0},
		RopeFrequencyBase: 10000, RopeScalingType: "linear", RopeScalingFactor: 4,
		RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var multiRoPE, outputBias, rmsNorm int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPEMulti {
			multiRoPE++
			attributes := node.Attrs.(tensor.RoPEMultiAttributes)
			if attributes.Sections != spec.RopeSections || attributes.FrequencyScale != 0.25 {
				t.Fatalf("unexpected %s MRoPE: %+v", architecture, attributes)
			}
		}
		if node.Op == tensor.OpAdd && len(node.Inputs) == 2 && node.Inputs[1].Name == "attn_output_bias" {
			outputBias++
		}
		if node.Op == tensor.OpRMSNorm {
			rmsNorm++
		}
	}
	if multiRoPE != 2 || outputBias != 1 || (architecture == "qwen3vl" && rmsNorm != 4) {
		t.Fatalf("%s graph has MRoPE=%d output-bias=%d RMS=%d", architecture, multiRoPE, outputBias, rmsNorm)
	}
}

func TestBuildDenseCommandR64UsesParallelResidualAndQKNorms(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "command-r", BlockCount: 64, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 8000000,
		LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardNorm = nil
	weights.FeedForwardNormBias = nil
	weights.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4, 2))
	weights.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4, 1))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNormCount, ropeCount int
	var attentionInput, feedForwardInput *tensor.Tensor
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNormCount++
		case tensor.OpRoPENormal:
			ropeCount++
		case tensor.OpMulMat:
			if node.Inputs[0].Name == "attn_q" {
				attentionInput = node.Inputs[1]
			}
			if node.Inputs[0].Name == "ffn_up" {
				feedForwardInput = node.Inputs[1]
			}
		}
	}
	if layerNormCount != 3 || ropeCount != 2 {
		t.Fatalf("Command R graph has LayerNorm=%d RoPE=%d, want 3/2", layerNormCount, ropeCount)
	}
	if attentionInput == nil || attentionInput != feedForwardInput {
		t.Fatal("Command R attention and FFN do not share the normalized block input")
	}
}

func TestBuildDensePLaMoUsesParallelResidualAndNeoX(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "plamo", BlockCount: 40, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10000,
		RMSNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardNorm = nil
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var rmsNormCount, ropeCount int
	var attentionInput, feedForwardInput *tensor.Tensor
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRMSNorm:
			rmsNormCount++
		case tensor.OpRoPENeoX:
			ropeCount++
		case tensor.OpMulMat:
			if node.Inputs[0].Name == "attn_q" {
				attentionInput = node.Inputs[1]
			}
			if node.Inputs[0].Name == "ffn_up" {
				feedForwardInput = node.Inputs[1]
			}
		}
	}
	if rmsNormCount != 1 || ropeCount != 2 {
		t.Fatalf("PLaMo graph has RMSNorm=%d RoPE=%d, want 1/2", rmsNormCount, ropeCount)
	}
	if attentionInput == nil || attentionInput != feedForwardInput {
		t.Fatal("PLaMo attention and FFN do not share the normalized block input")
	}
}

func TestBuildDensePLaMo3UsesPreRoPEQKNormPostNormAndFusedSwiGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "plamo3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		RMSNormEpsilon: 1e-6, SlidingWindow: 128, SlidingPattern: 2,
		LayerHeadCounts: []uint32{2, 4}, LayerKVHeadCounts: []uint32{1, 2},
		LayerFeedForward: []uint32{12, 16},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:        builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:      builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(2)),
		AttentionKNorm:      builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(2)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(4, 8)),
		AttentionPostNorm:   builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 24)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardPostNorm: builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var rmsNorms, groupSlices int
	var attention *tensor.Tensor
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRMSNorm:
			rmsNorms++
		case tensor.OpGroupSlice:
			groupSlices++
		case tensor.OpAttention:
			attention = node
		}
		if node.Op == tensor.OpRoPENeoX {
			attributes := node.Attrs.(tensor.RoPEAttributes)
			if attributes.FrequencyBase != 20000 || node.Inputs[0].Op != tensor.OpMultiply ||
				node.Inputs[0].Inputs[0].Op != tensor.OpRMSNorm {
				t.Fatalf("PLaMo 3 NeoX RoPE does not follow Q/K norm: %+v", node)
			}
		}
	}
	if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Window != 128 ||
		rmsNorms != 6 || groupSlices != 5 {
		t.Fatalf("PLaMo 3 graph: attention=%+v RMSNorm=%d slices=%d", attention, rmsNorms, groupSlices)
	}
}

func TestBuildDenseStableLMOptionalNormLayouts(t *testing.T) {
	for _, sequential := range []bool{false, true} {
		builder := tensor.NewBuilder()
		spec := Spec{
			Architecture: "stablelm", BlockCount: 40, EmbeddingLength: 8,
			FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
			RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
		}
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := denseBlockInputs(builder, spec)
		if sequential {
			weights.AttentionQNorm = nil
			weights.AttentionKNorm = nil
		} else {
			weights.FeedForwardNorm = nil
			weights.FeedForwardNormBias = nil
			weights.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4, 2))
			weights.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4, 1))
		}
		output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := tensor.Topological(output)
		if err != nil {
			t.Fatal(err)
		}
		var layerNormCount, ropeCount int
		var attentionInput, feedForwardInput *tensor.Tensor
		for _, node := range nodes {
			switch node.Op {
			case tensor.OpLayerNorm:
				layerNormCount++
			case tensor.OpRoPENeoX:
				ropeCount++
				if node.Attrs.(tensor.RoPEAttributes).RotaryDimensions != 2 {
					t.Fatal("StableLM partial rotary dimension was not honored")
				}
			case tensor.OpMulMat:
				if node.Inputs[0].Name == "attn_q" {
					attentionInput = node.Inputs[1]
				}
				if node.Inputs[0].Name == "ffn_up" {
					feedForwardInput = node.Inputs[1]
				}
			}
		}
		wantNorms := 3
		if sequential {
			wantNorms = 2
		}
		if layerNormCount != wantNorms || ropeCount != 2 {
			t.Fatalf("StableLM graph has LayerNorm=%d RoPE=%d", layerNormCount, ropeCount)
		}
		if (attentionInput == feedForwardInput) == sequential {
			t.Fatal("StableLM residual topology does not match its FFN norm layout")
		}
	}
}

func TestBuildDensePhi2FusedQKVAndQueryScale(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "phi2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(24))
	weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardNorm = nil
	weights.FeedForwardNormBias = nil
	weights.FeedForwardGate = nil
	weights.FeedForwardUpBias = builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var slices, ropeCount, geluCount int
	var attentionScale, queryScale float32
	var attentionInput, feedForwardInput *tensor.Tensor
	offsets := map[uint64]bool{}
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupSlice:
			slices++
			offsets[node.Attrs.(tensor.GroupSliceAttributes).Offset] = true
		case tensor.OpRoPENeoX:
			ropeCount++
			if node.Attrs.(tensor.RoPEAttributes).RotaryDimensions != 2 {
				t.Fatal("Phi-2 partial rotary dimension was not honored")
			}
		case tensor.OpGELU:
			geluCount++
		case tensor.OpScale:
			queryScale = node.Attrs.(tensor.ScaleAttributes).Value
		case tensor.OpAttention:
			attentionScale = node.Attrs.(tensor.AttentionAttributes).Scale
		case tensor.OpMulMat:
			if node.Inputs[0].Name == "attn_qkv" {
				attentionInput = node.Inputs[1]
			}
			if node.Inputs[0].Name == "ffn_up" {
				feedForwardInput = node.Inputs[1]
			}
		}
	}
	if slices != 3 || !offsets[0] || !offsets[8] || !offsets[16] ||
		ropeCount != 2 || geluCount != 1 || queryScale != 0.5 || attentionScale != 1 {
		t.Fatalf(
			"unexpected Phi-2 graph: slices=%d offsets=%v rope=%d GELU=%d qscale=%v ascale=%v",
			slices, offsets, ropeCount, geluCount, queryScale, attentionScale,
		)
	}
	if attentionInput == nil || attentionInput != feedForwardInput {
		t.Fatal("Phi-2 attention and FFN do not share the normalized block input")
	}
}

func TestBuildDensePhi3FusedProjectionsAndLongRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "phi3", BlockCount: 1, ContextLength: 128,
		OriginalContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		RopeScalingType: "longrope", RopeAttentionFactor: 1.25, RMSNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(24))
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 24))
	weights.RopeFactors = builder.Input("rope_long", dtype.F32, tensor.MustShape(2))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var slices, silu, ropeWithFactors, queryPrescale, ropeScale int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupSlice:
			slices++
		case tensor.OpSiLU:
			silu++
		case tensor.OpRoPENeoX:
			if len(node.Inputs) == 2 && node.Inputs[1] == weights.RopeFactors {
				ropeWithFactors++
			}
		case tensor.OpScale:
			value := node.Attrs.(tensor.ScaleAttributes).Value
			if value == 0.5 {
				queryPrescale++
			}
			if value == 1.25 {
				ropeScale++
			}
		}
	}
	if slices != 5 || silu != 1 || ropeWithFactors != 2 ||
		queryPrescale != 1 || ropeScale != 2 {
		t.Fatalf("unexpected Phi-3 graph: slices=%d SiLU=%d rope=%d qscale=%d rope-scale=%d", slices, silu, ropeWithFactors, queryPrescale, ropeScale)
	}
}

func TestBuildDensePhiMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "phimoe", BlockCount: 1, ContextLength: 128,
		OriginalContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 12,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		RopeScalingType: "longrope", RopeAttentionFactor: 1.25, RMSNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	attentionNormBias := builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8))
	feedForwardNormBias := builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(8))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:      attentionNormBias,
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias:    builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNormBias:    feedForwardNormBias,
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
		RopeFactors:            builder.Input("rope_long", dtype.F32, tensor.MustShape(2)),
	}
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var normBiasAdds, ropeWithFactors, queryPrescale int
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			moe = node
		}
		if node.Op == tensor.OpAdd && len(node.Inputs) == 2 &&
			(node.Inputs[1] == attentionNormBias || node.Inputs[1] == feedForwardNormBias) {
			normBiasAdds++
		}
		if node.Op == tensor.OpRoPENeoX && len(node.Inputs) == 2 && node.Inputs[1] == weights.RopeFactors {
			ropeWithFactors++
		}
		if node.Op == tensor.OpScale && node.Attrs.(tensor.ScaleAttributes).Value == 0.5 {
			queryPrescale++
		}
	}
	if moe == nil || !moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb ||
		normBiasAdds != 2 || ropeWithFactors != 2 || queryPrescale != 1 {
		t.Fatalf("unexpected PhiMoE graph: moe=%v norm-bias=%d rope=%d qscale=%d", moe != nil, normBiasAdds, ropeWithFactors, queryPrescale)
	}
}

func TestBuildDenseEXAOneMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "exaone-moe", BlockCount: 4, EmbeddingLength: 8, FeedForwardLength: 16,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1.5,
		SharedExpertFF: 12, ExpertGatingFunc: 2, ExpertWeightsNorm: true,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 500000,
		SlidingWindow: 128, SlidingPattern: 4, RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	local, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(local.Output)
	var moe *tensor.Tensor
	var rope, window, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
		case tensor.OpAttention:
			if node.Attrs.(tensor.AttentionAttributes).Window > 0 {
				window++
			}
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil || moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		len(moe.Inputs) != 7 || rope != 2 || window != 1 || silu < 1 {
		t.Fatalf("unexpected EXAONE-MoE graph: moe=%v rope=%d window=%d silu=%d", moe != nil, rope, window, silu)
	}
}

func TestBuildSmallThinkerUsesSplitRouterReGLUAndSlidingRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "smallthinker", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 6, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25, ExpertGatingFunc: 2,
		ExpertWeightsNorm: true, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128, SlidingPattern: 4, NoRopeLayerStep: 4,
		RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var rope, sliding int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
		case tensor.OpAttention:
			if node.Attrs.(tensor.AttentionAttributes).Window == 128 {
				sliding++
			}
		}
	}
	if moe == nil {
		t.Fatal("SmallThinker graph is missing MoE")
	}
	attributes := moe.Attrs.(tensor.MoEAttributes)
	if attributes.Activation != tensor.MoEActivationReLU ||
		attributes.Routing != tensor.MoERoutingSigmoid || !attributes.NormalizeTopKProb ||
		len(moe.Inputs) != 6 || moe.Inputs[1] != input || moe.Inputs[0] == input ||
		rope != 2 || sliding != 1 {
		t.Fatalf("unexpected SmallThinker graph: attrs=%+v inputs=%d rope=%d sliding=%d", attributes, len(moe.Inputs), rope, sliding)
	}
}
