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
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(embedding)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(embedding, query)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(embedding, keyValue)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(embedding, value)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(attentionOutput, embedding)),
		AttentionQNorm:  builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(uint64(spec.KeyLength))),
		AttentionKNorm:  builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(uint64(spec.KeyLength))),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(feedForward, embedding)),
	}
}
