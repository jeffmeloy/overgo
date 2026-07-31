package model

import (
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

func TestBuildDenseQwen3Block(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "qwen3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(input.Shape) {
		t.Fatalf("output shape = %v, want %v", output.Shape.Slice(), input.Shape.Slice())
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var foundAttention bool
	for _, node := range nodes {
		foundAttention = foundAttention || node.Op == tensor.OpAttention
	}
	if !foundAttention {
		t.Fatal("dense block graph has no attention operation")
	}
}

func TestBuildDenseQwen2BlockUsesNeoXAndProjectionBiases(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "qwen2",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.AttentionQBias = builder.Input("attn_q.bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionKBias = builder.Input("attn_k.bias", dtype.F32, tensor.MustShape(4))
	weights.AttentionVBias = builder.Input("attn_v.bias", dtype.F32, tensor.MustShape(4))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var neoxCount int
	found := map[string]bool{}
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENeoX {
			neoxCount++
		}
		found[node.Name] = true
	}
	if neoxCount != 2 {
		t.Fatalf("Qwen 2 NeoX RoPE count = %d, want 2", neoxCount)
	}
	for _, name := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias"} {
		if !found[name] {
			t.Fatalf("Qwen 2 projection bias %q is disconnected", name)
		}
	}
}

func TestBuildDenseInternLM2EXAONEAndXVERSEUseExpectedRoPE(t *testing.T) {
	for _, test := range []struct {
		architecture string
		want         tensor.Op
	}{
		{"internlm2", tensor.OpRoPENormal},
		{"exaone", tensor.OpRoPENeoX},
		{"mistral3", tensor.OpRoPENormal},
		{"xverse", tensor.OpRoPENormal},
	} {
		t.Run(test.architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{
				Architecture:      test.architecture,
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
			weights.AttentionQNorm = nil
			weights.AttentionKNorm = nil
			if test.architecture == "exaone" {
				weights.RopeFactors = builder.Input(
					"rope_factors",
					dtype.F32,
					tensor.MustShape(2),
				)
			}
			output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(output)
			if err != nil {
				t.Fatal(err)
			}
			var ropeCount int
			for _, node := range nodes {
				if node.Op == test.want {
					ropeCount++
					if weights.RopeFactors != nil &&
						(len(node.Inputs) != 2 || node.Inputs[1] != weights.RopeFactors) {
						t.Fatal("EXAONE RoPE factors are disconnected")
					}
				} else if node.Op == tensor.OpRoPENormal || node.Op == tensor.OpRoPENeoX {
					t.Fatalf("unexpected RoPE operation %s", node.Op)
				}
			}
			if ropeCount != 2 {
				t.Fatalf("%s RoPE count = %d, want 2", test.architecture, ropeCount)
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

func TestBuildDenseQwen3BlockWithCache(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "qwen3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 1))
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(4, 1, 3))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(4, 1, 3))
	result, err := BuildDenseBlockCached(
		builder,
		input,
		spec,
		denseBlockInputs(builder, spec),
		[]uint32{3},
		pastKey,
		pastValue,
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Key.Shape.Dims[2] != 4 || result.Value.Shape.Dims[2] != 4 {
		t.Fatalf("cache token counts = %d/%d, want 4/4", result.Key.Shape.Dims[2], result.Value.Shape.Dims[2])
	}
	if !result.Output.Shape.Equal(input.Shape) {
		t.Fatalf("output shape = %v, want %v", result.Output.Shape.Slice(), input.Shape.Slice())
	}
}

func TestBuildDenseLlamaBlockUsesNormalRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.RopeFactors = builder.Input("rope_factors", dtype.F32, tensor.MustShape(2))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	normalCount := 0
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENormal {
			normalCount++
			if len(node.Inputs) != 2 || node.Inputs[1] != weights.RopeFactors {
				t.Fatal("Llama RoPE node does not consume frequency factors")
			}
		}
		if node.Op == tensor.OpRoPENeoX {
			t.Fatal("Llama block unexpectedly uses NeoX RoPE")
		}
	}
	if normalCount != 2 {
		t.Fatalf("normal RoPE count = %d, want 2", normalCount)
	}
}

func TestBuildDenseLlamaBlockUsesLinearRoPEScale(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeScalingType:   "linear",
		RopeScalingFactor: 8,
		RMSNormEpsilon:    1e-5,
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
	ropeCount := 0
	for _, node := range nodes {
		if node.Op != tensor.OpRoPENormal {
			continue
		}
		ropeCount++
		attributes := node.Attrs.(tensor.RoPEAttributes)
		if attributes.FrequencyScale != 0.125 {
			t.Fatalf("RoPE frequency scale = %v, want 0.125", attributes.FrequencyScale)
		}
	}
	if ropeCount != 2 {
		t.Fatalf("normal RoPE count = %d, want 2", ropeCount)
	}
}

func TestBuildDenseGemma2BlockUsesSoftcappedSlidingAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "gemma2",
		BlockCount:        26,
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
		AttentionSoftcap:  50,
		RMSNormEpsilon:    1e-6,
		SlidingWindow:     4096,
		SlidingPattern:    2,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.AttentionPostNorm = builder.Input(
		"post_attention_norm", dtype.F32, tensor.MustShape(8),
	)
	weights.FeedForwardPostNorm = builder.Input(
		"post_ffw_norm", dtype.F32, tensor.MustShape(8),
	)
	output, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attentionCount, neoxCount int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attentionCount++
			attributes := node.Attrs.(tensor.AttentionAttributes)
			if attributes.Softcap != 50 || attributes.Window != 4096 || attributes.Scale != 1 {
				t.Fatalf("unexpected Gemma 2 attention attributes: %+v", attributes)
			}
		case tensor.OpRoPENeoX:
			neoxCount++
			attributes := node.Attrs.(tensor.RoPEAttributes)
			if attributes.FrequencyScale != 0.25 {
				t.Fatalf("Gemma 2 sliding RoPE scale = %v, want 0.25", attributes.FrequencyScale)
			}
		}
	}
	if attentionCount != 1 || neoxCount != 2 {
		t.Fatalf("Gemma 2 graph has attention=%d NeoX RoPE=%d, want 1/2", attentionCount, neoxCount)
	}
}

func TestBuildDenseGemmaBlockUsesScaledNeoXGEGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "gemma",
		BlockCount:        18,
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
	var neoxCount, scaleCount, geluCount int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRoPENeoX:
			neoxCount++
		case tensor.OpScale:
			scaleCount++
			if value := node.Attrs.(tensor.ScaleAttributes).Value; value != 0.5 {
				t.Fatalf("Gemma query scale = %v, want 0.5", value)
			}
		case tensor.OpGELU:
			geluCount++
		}
	}
	if neoxCount != 2 || scaleCount != 1 || geluCount != 1 {
		t.Fatalf(
			"Gemma graph has NeoX=%d scale=%d GELU=%d, want 2/1/1",
			neoxCount,
			scaleCount,
			geluCount,
		)
	}
}

func TestBuildDenseBlockConsumesProjectionBiases(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.AttentionQBias = builder.Input("attn_q.bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionKBias = builder.Input("attn_k.bias", dtype.F32, tensor.MustShape(4))
	weights.AttentionVBias = builder.Input("attn_v.bias", dtype.F32, tensor.MustShape(4))
	weights.AttentionOutputBias = builder.Input("attn_output.bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardGateBias = builder.Input("ffn_gate.bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardUpBias = builder.Input("ffn_up.bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("ffn_down.bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	found := make(map[string]bool)
	for _, node := range nodes {
		found[node.Name] = true
	}
	for _, name := range []string{
		"attn_q.bias",
		"attn_k.bias",
		"attn_v.bias",
		"attn_output.bias",
		"ffn_gate.bias",
		"ffn_up.bias",
		"ffn_down.bias",
	} {
		if !found[name] {
			t.Fatalf("projection bias %q is not connected to the graph", name)
		}
	}
}

func TestBuildQwen35AttentionBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := qwen35TestSpec()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	result, err := BuildQwen35BlockCached(
		builder,
		input,
		spec,
		qwen35AttentionInputs(builder, spec),
		[]uint32{0, 1},
		false,
		nil,
		nil,
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) ||
		!result.Key.Shape.Equal(tensor.MustShape(4, 1, 2)) ||
		!result.Value.Shape.Equal(tensor.MustShape(4, 1, 2)) {
		t.Fatalf("unexpected Qwen3.5 attention result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	var groupSlices, multiRoPE, sigmoid int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupSlice:
			groupSlices++
		case tensor.OpRoPEMulti:
			multiRoPE++
		case tensor.OpSigmoid:
			sigmoid++
		}
	}
	if groupSlices != 2 || multiRoPE != 2 || sigmoid != 1 {
		t.Fatalf(
			"Qwen3.5 attention ops: group slices=%d RoPE=%d sigmoid=%d",
			groupSlices,
			multiRoPE,
			sigmoid,
		)
	}
}

func TestBuildT5EncoderBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "t5encoder",
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       2,
		KeyLength:         4,
		ValueLength:       4,
		RMSNormEpsilon:    1e-6,
		RelativeBuckets:   4,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionNorm:         builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:            builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:            builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:            builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:       builder.Input("attn_o", dtype.F32, tensor.MustShape(8, 8)),
		AttentionRelativeBias: builder.Input("attn_rel_b", dtype.F32, tensor.MustShape(2, 4)),
		FeedForwardNorm:       builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:       builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUp:         builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown:       builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	output, err := BuildT5EncoderBlock(builder, input, spec, weights)
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(input.Shape) {
		t.Fatalf("T5 output shape = %v, want %v", output.Shape, input.Shape)
	}
	var foundRelativeAttention, foundGELU bool
	order, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range order {
		if node.Op == tensor.OpAttention {
			attributes := node.Attrs.(tensor.AttentionAttributes)
			foundRelativeAttention = len(node.Inputs) == 4 &&
				attributes.RelativeBuckets == 4 &&
				!attributes.Causal
		}
		foundGELU = foundGELU || node.Op == tensor.OpGELU
	}
	if !foundRelativeAttention || !foundGELU {
		t.Fatalf(
			"T5 graph relative attention/GELU = %t/%t",
			foundRelativeAttention,
			foundGELU,
		)
	}
}

func TestBuildQwen35RecurrentBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := qwen35TestSpec()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
	result, err := BuildQwen35BlockCached(
		builder,
		input,
		spec,
		qwen35RecurrentInputs(builder, spec),
		[]uint32{0, 1},
		true,
		nil,
		nil,
		convState,
		ssmState,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) ||
		!result.ConvState.Shape.Equal(convState.Shape) ||
		!result.SSMState.Shape.Equal(ssmState.Shape) ||
		!result.Recurrent {
		t.Fatalf("unexpected Qwen3.5 recurrent result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.ConvState, result.SSMState)
	if err != nil {
		t.Fatal(err)
	}
	var conv, delta int
	for _, node := range nodes {
		if node.Op == tensor.OpSSMConv {
			conv++
		}
		if node.Op == tensor.OpGatedDeltaNet {
			delta++
		}
	}
	if conv != 1 || delta != 1 {
		t.Fatalf("Qwen3.5 recurrent ops: convolution=%d delta-net=%d", conv, delta)
	}
}

func qwen35TestSpec() Spec {
	return Spec{
		Architecture:          "qwen35",
		EmbeddingLength:       8,
		FeedForwardLength:     12,
		HeadCount:             2,
		HeadCountKV:           1,
		KeyLength:             4,
		ValueLength:           4,
		RopeFrequencyBase:     10000,
		RMSNormEpsilon:        1e-6,
		RopeDimensionCount:    4,
		RopeSections:          [4]int32{1, 1, 0, 0},
		SSMConvKernel:         3,
		SSMInnerSize:          4,
		SSMStateSize:          2,
		SSMTimeStepRank:       2,
		SSMGroupCount:         1,
		FullAttentionInterval: 4,
	}
}

func qwen35CommonInputs(builder *tensor.Builder, spec Spec) LayerGraphWeights {
	embedding := uint64(spec.EmbeddingLength)
	feedForward := uint64(spec.FeedForwardLength)
	return LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardNorm: builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(feedForward, embedding)),
	}
}

func qwen35AttentionInputs(builder *tensor.Builder, spec Spec) LayerGraphWeights {
	result := qwen35CommonInputs(builder, spec)
	embedding := uint64(spec.EmbeddingLength)
	headWidth := uint64(spec.KeyLength)
	result.AttentionQ = builder.Input(
		"attn_q",
		dtype.F32,
		tensor.MustShape(embedding, 2*uint64(spec.HeadCount)*headWidth),
	)
	result.AttentionK = builder.Input(
		"attn_k",
		dtype.F32,
		tensor.MustShape(embedding, uint64(spec.HeadCountKV)*headWidth),
	)
	result.AttentionV = builder.Input(
		"attn_v",
		dtype.F32,
		tensor.MustShape(embedding, uint64(spec.HeadCountKV)*uint64(spec.ValueLength)),
	)
	result.AttentionOutput = builder.Input("attn_output", dtype.F32, tensor.MustShape(embedding, embedding))
	result.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(headWidth))
	result.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(headWidth))
	return result
}

func qwen35RecurrentInputs(builder *tensor.Builder, spec Spec) LayerGraphWeights {
	result := qwen35CommonInputs(builder, spec)
	embedding := uint64(spec.EmbeddingLength)
	state := uint64(spec.SSMStateSize)
	keyDimension := state * uint64(spec.SSMGroupCount)
	valueDimension := uint64(spec.SSMInnerSize)
	channels := 2*keyDimension + valueDimension
	valueHeads := uint64(spec.SSMTimeStepRank)
	result.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(embedding, channels))
	result.AttentionGate = builder.Input("attn_gate", dtype.F32, tensor.MustShape(embedding, valueDimension))
	result.SSMConv1D = builder.Input(
		"ssm_conv1d",
		dtype.F32,
		tensor.MustShape(uint64(spec.SSMConvKernel), channels),
	)
	result.SSMTimeStep = builder.Input("ssm_dt", dtype.F32, tensor.MustShape(valueHeads))
	result.SSMA = builder.Input("ssm_a", dtype.F32, tensor.MustShape(valueHeads))
	result.SSMBeta = builder.Input("ssm_beta", dtype.F32, tensor.MustShape(embedding, valueHeads))
	result.SSMAlpha = builder.Input("ssm_alpha", dtype.F32, tensor.MustShape(embedding, valueHeads))
	result.SSMNorm = builder.Input("ssm_norm", dtype.F32, tensor.MustShape(state))
	result.SSMOutput = builder.Input("ssm_out", dtype.F32, tensor.MustShape(valueDimension, embedding))
	return result
}

func denseBlockInputs(builder *tensor.Builder, spec Spec) LayerGraphWeights {
	embedding := uint64(spec.EmbeddingLength)
	feedForward := uint64(spec.FeedForwardLength)
	query := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	keyValue := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	value := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	attentionOutput := uint64(spec.HeadCount) * uint64(spec.ValueLength)
	return LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(embedding)),
		AttentionNormBias:   builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(embedding)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(embedding, query)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(embedding, keyValue)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(embedding, value)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(attentionOutput, embedding)),
		AttentionQNorm:      builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(uint64(spec.KeyLength))),
		AttentionKNorm:      builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(uint64(spec.KeyLength))),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardNormBias: builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(feedForward, embedding)),
	}
}
