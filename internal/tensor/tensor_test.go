package tensor

import (
	"math"
	"strings"
	"testing"

	"llamacpp2go/internal/tensor/dtype"
)

func TestBuilderAndTopological(t *testing.T) {
	builder := NewBuilder()
	shape := MustShape(4, 2)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.SiLU(builder.Scale(builder.Add(left, right), 0.5))
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	nodes, err := Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != 5 || nodes[len(nodes)-1] != output {
		t.Fatalf("unexpected topological order: %#v", nodes)
	}
}

func TestBuilderRejectsShapeMismatch(t *testing.T) {
	builder := NewBuilder()
	left := builder.Input("left", dtype.F32, MustShape(4))
	right := builder.Input("right", dtype.F32, MustShape(8))
	if result := builder.Add(left, right); result != nil {
		t.Fatal("shape-mismatched add returned a tensor")
	}
	if err := builder.Err(); err == nil || !strings.Contains(err.Error(), "cannot broadcast") {
		t.Fatalf("error = %v", err)
	}
}

func TestBuilderAttentionSinks(t *testing.T) {
	builder := NewBuilder()
	query := builder.Input("query", dtype.F32, MustShape(2, 2, 1))
	key := builder.Input("key", dtype.F32, MustShape(2, 1, 1))
	value := builder.Input("value", dtype.F32, MustShape(2, 1, 1))
	sinks := builder.Input("sinks", dtype.F32, MustShape(2))
	output := builder.AttentionWithSinksWithOffset(query, key, value, sinks, 1, true, 0)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(AttentionAttributes)
	if !attributes.HasSinks || len(output.Inputs) != 4 || output.Inputs[3] != sinks {
		t.Fatalf("unexpected sink attention: %+v", attributes)
	}

	invalid := NewBuilder()
	invalid.AttentionWithSinksWithOffset(
		invalid.Input("query", dtype.F32, MustShape(2, 2, 1)),
		invalid.Input("key", dtype.F32, MustShape(2, 1, 1)),
		invalid.Input("value", dtype.F32, MustShape(2, 1, 1)),
		invalid.Input("sinks", dtype.F32, MustShape(1)), 1, true, 0,
	)
	if invalid.Err() == nil || !strings.Contains(invalid.Err().Error(), "[query heads]") {
		t.Fatalf("error = %v", invalid.Err())
	}
}

func TestBuilderBroadcastWeightedNormAndSwiGLU(t *testing.T) {
	builder := NewBuilder()
	activation := builder.Input("activation", dtype.F32, MustShape(4, 3))
	weight := builder.Input("weight", dtype.F32, MustShape(4))
	norm := builder.WeightedRMSNorm(activation, weight, 1e-5)
	up := builder.Input("up", dtype.F32, MustShape(4, 3))
	output := builder.SwiGLU(norm, up)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(MustShape(4, 3)) {
		t.Fatalf("output shape = %v", output.Shape.Slice())
	}
	if norm.Op != OpMultiply || output.Op != OpMultiply {
		t.Fatalf("unexpected composed ops: %s and %s", norm.Op, output.Op)
	}
}

func TestBuilderAffineLayerNorm(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(4, 2))
	weight := builder.Input("weight", dtype.F32, MustShape(4))
	bias := builder.Input("bias", dtype.F32, MustShape(4))
	output := builder.AffineLayerNorm(input, weight, bias, 1e-5)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if output.Op != OpAdd ||
		output.Inputs[0].Op != OpMultiply ||
		output.Inputs[0].Inputs[0].Op != OpLayerNorm {
		t.Fatalf("unexpected affine LayerNorm graph ending in %s", output.Op)
	}

	invalid := NewBuilder()
	invalid.LayerNorm(invalid.Input("input", dtype.F32, MustShape(1)), 0)
	if invalid.Err() == nil {
		t.Fatal("zero LayerNorm epsilon was accepted")
	}
}

func TestBuilderQwen35UnaryPrimitives(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(4, 3))
	output := builder.L2Norm(builder.Softplus(builder.Sigmoid(input)), 1e-6)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if output.Op != OpL2Norm || output.Inputs[0].Op != OpSoftplus ||
		output.Inputs[0].Inputs[0].Op != OpSigmoid {
		t.Fatalf("unexpected unary graph ending in %s", output.Op)
	}

	invalid := NewBuilder()
	invalid.L2Norm(invalid.Input("input", dtype.F32, MustShape(1)), -1)
	if invalid.Err() == nil {
		t.Fatal("negative L2Norm epsilon was accepted")
	}
}

func TestBuilderXIELU(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(4, 3))
	output := builder.XIELU(input, 0.8, 0.2, 0.5, -1e-6)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes, ok := output.Attrs.(XIELUAttributes)
	if output.Op != OpXIELU || !ok || attributes.AlphaN != 0.8 || attributes.Epsilon != -1e-6 {
		t.Fatalf("unexpected xIELU node: %#v", output)
	}

	invalid := NewBuilder()
	invalid.XIELU(invalid.Input("input", dtype.F32, MustShape(1)), float32(math.NaN()), 0, 0, 0)
	if invalid.Err() == nil {
		t.Fatal("non-finite xIELU parameter was accepted")
	}
}

func TestBuilderSSMConv(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(6, 4, 2))
	weights := builder.Input("weights", dtype.F32, MustShape(3, 4))
	output := builder.SSMConv(input, weights)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(MustShape(4, 4, 2)) {
		t.Fatalf("SSMConv output shape = %v", output.Shape.Slice())
	}
}

func TestBuilderRepeatHeads(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1, 3))
	output := builder.RepeatHeads(input, 4)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(MustShape(2, 4, 3)) {
		t.Fatalf("RepeatHeads output shape = %v", output.Shape.Slice())
	}

	invalid := NewBuilder()
	invalid.RepeatHeads(invalid.Input("input", dtype.F32, MustShape(2, 2, 3)), 4)
	if invalid.Err() == nil {
		t.Fatal("RepeatHeads accepted multiple input heads")
	}
}

func TestBuilderSigmoidMoEWithSelectionBias(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	gate := builder.Input("gate", dtype.F32, MustShape(2, 4, 3))
	up := builder.Input("up", dtype.F32, MustShape(2, 4, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	bias := builder.Input("bias", dtype.F32, MustShape(3))
	output := builder.MoESigmoid(input, router, gate, up, down, bias, 2, true, 1.5)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if attributes.Routing != MoERoutingSigmoid || attributes.Activation != MoEActivationSiLU ||
		len(output.Inputs) != 7 {
		t.Fatalf("unexpected sigmoid MoE graph: %+v", output)
	}
}

func TestBuilderLimitedSigmoidMoE(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	gate := builder.Input("gate", dtype.F32, MustShape(2, 4, 3))
	up := builder.Input("up", dtype.F32, MustShape(2, 4, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	output := builder.MoESigmoidLimited(input, router, gate, up, down, nil, 2, true, 1, 3)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if attributes.SwiGLUClamp != 3 || attributes.Routing != MoERoutingSigmoid {
		t.Fatalf("unexpected limited MoE attributes: %+v", attributes)
	}

	invalid := NewBuilder()
	x := invalid.Input("input", dtype.F32, MustShape(2, 1))
	r := invalid.Input("router", dtype.F32, MustShape(2, 3))
	g := invalid.Input("gate", dtype.F32, MustShape(2, 4, 3))
	u := invalid.Input("up", dtype.F32, MustShape(2, 4, 3))
	d := invalid.Input("down", dtype.F32, MustShape(4, 2, 3))
	invalid.MoESigmoidLimited(x, r, g, u, d, nil, 2, true, 1, -1)
	if invalid.Err() == nil {
		t.Fatal("limited MoE accepted negative clamp")
	}
}

func TestBuilderSigmoidMoEWithFusedGateUp(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	gateUp := builder.Input("gate_up", dtype.F32, MustShape(2, 8, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	output := builder.MoESigmoidFusedGateUp(input, router, gateUp, down, nil, 2, true, 1.5)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if !attributes.Gated || !attributes.FusedGateUp || attributes.Routing != MoERoutingSigmoid ||
		len(output.Inputs) != 5 {
		t.Fatalf("unexpected fused sigmoid MoE graph: %+v", output)
	}
}

func TestBuilderSoftmaxMoEWithFusedGateUpAndSelectionBias(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	gateUp := builder.Input("gate_up", dtype.F32, MustShape(2, 8, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	bias := builder.Input("bias", dtype.F32, MustShape(3))
	output := builder.MoESoftmaxFusedGateUp(input, router, gateUp, down, bias, 2, true, 1.5)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if !attributes.Gated || !attributes.FusedGateUp || attributes.Routing != MoERoutingSoftmax ||
		len(output.Inputs) != 6 {
		t.Fatalf("unexpected fused softmax MoE graph: %+v", output)
	}
}

func TestBuilderSoftmaxMoEWithSelectionBias(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	gate := builder.Input("gate", dtype.F32, MustShape(2, 4, 3))
	up := builder.Input("up", dtype.F32, MustShape(2, 4, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	bias := builder.Input("bias", dtype.F32, MustShape(3))
	output := builder.MoESoftmaxWithSelectionBias(input, router, gate, up, down, bias, 2, true, 1.5)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if attributes.Routing != MoERoutingSoftmax || attributes.Activation != MoEActivationSiLU ||
		len(output.Inputs) != 7 {
		t.Fatalf("unexpected softmax MoE graph: %+v", output)
	}
}

func TestBuilderUngatedMoE(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	up := builder.Input("up", dtype.F32, MustShape(2, 4, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	output := builder.MoEUngated(input, router, up, down, 2, true, 1.5)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if attributes.Gated || attributes.Routing != MoERoutingSoftmax ||
		attributes.Activation != MoEActivationSiLU || len(output.Inputs) != 5 {
		t.Fatalf("unexpected ungated MoE graph: %+v", output)
	}

	quantized := NewBuilder()
	quantizedInput := quantized.Input("input", dtype.F32, MustShape(32, 1))
	quantizedRouter := quantized.Input("router", dtype.F32, MustShape(32, 3))
	quantizedUp := quantized.Input("up", dtype.Q8_0, MustShape(32, 32, 3))
	quantizedDown := quantized.Input("down", dtype.Q8_0, MustShape(32, 32, 3))
	quantized.MoEUngated(quantizedInput, quantizedRouter, quantizedUp, quantizedDown, 2, true, 1)
	if err := quantized.Err(); err != nil {
		t.Fatal(err)
	}
}

func TestBuilderUngatedMoEWithSelectionBias(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	up := builder.Input("up", dtype.F32, MustShape(2, 4, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	bias := builder.Input("bias", dtype.F32, MustShape(3))
	output := builder.MoEUngatedWithSelectionBias(input, router, up, down, bias, 2, true, 1.5)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if attributes.Gated || attributes.Routing != MoERoutingSoftmax ||
		attributes.Activation != MoEActivationSiLU || !attributes.NormalizeTopKProb ||
		attributes.Scale != 1.5 || len(output.Inputs) != 6 {
		t.Fatalf("unexpected biased ungated MoE graph: %+v", output)
	}
}

func TestBuilderReLUMoEWithRouterInput(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	routerInput := builder.Input("router-input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	gate := builder.Input("gate", dtype.F32, MustShape(2, 4, 3))
	up := builder.Input("up", dtype.F32, MustShape(2, 4, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	output := builder.MoEReLUWithRouterInput(
		input, routerInput, router, gate, up, down, 2, true, 1, MoERoutingSigmoid,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if attributes.Activation != MoEActivationReLU || attributes.Routing != MoERoutingSigmoid ||
		output.Inputs[1] != routerInput || len(output.Inputs) != 6 {
		t.Fatalf("unexpected ReLU MoE graph: %+v", output)
	}
}

func TestBuilderGroupedMoEWithRouterInput(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	routerInput := builder.Input("router-input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 4))
	gate := builder.Input("gate", dtype.F32, MustShape(2, 3, 2))
	up := builder.Input("up", dtype.F32, MustShape(2, 3, 2))
	down := builder.Input("down", dtype.F32, MustShape(3, 2, 2))
	output := builder.MoEGroupedWithRouterInput(
		input, routerInput, router, gate, up, down, 2, true, 1.25, 2,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if attributes.Experts != 4 || attributes.ExpertIndexDivisor != 2 ||
		output.Inputs[1] != routerInput || len(output.Inputs) != 6 {
		t.Fatalf("unexpected grouped MoE graph: %+v", output)
	}
	invalid := NewBuilder()
	invalidInput := invalid.Input("input", dtype.F32, MustShape(2, 1))
	invalidRouter := invalid.Input("router", dtype.F32, MustShape(2, 4))
	invalidExperts := invalid.Input("experts", dtype.F32, MustShape(2, 3, 2))
	invalidDown := invalid.Input("down", dtype.F32, MustShape(3, 2, 2))
	invalid.MoEGroupedWithRouterInput(
		invalidInput, invalidInput, invalidRouter, invalidExperts, invalidExperts,
		invalidDown, 2, true, 1, 0,
	)
	if invalid.Err() == nil {
		t.Fatal("grouped MoE accepted zero expert index divisor")
	}
}

func TestBuilderGELUMoE(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(2, 1))
	router := builder.Input("router", dtype.F32, MustShape(2, 3))
	gate := builder.Input("gate", dtype.F32, MustShape(2, 4, 3))
	up := builder.Input("up", dtype.F32, MustShape(2, 4, 3))
	down := builder.Input("down", dtype.F32, MustShape(4, 2, 3))
	output := builder.MoEGELU(input, router, gate, up, down, 2, true, 1)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(MoEAttributes)
	if attributes.Activation != MoEActivationGELU || !attributes.Gated ||
		attributes.Routing != MoERoutingSoftmax || len(output.Inputs) != 6 {
		t.Fatalf("unexpected GELU MoE graph: %+v", output)
	}
}

func TestBuilderClamp(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(3))
	output := builder.Clamp(input, -2, 3)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	attributes := output.Attrs.(ClampAttributes)
	if output.Op != OpClamp || attributes.Minimum != -2 || attributes.Maximum != 3 {
		t.Fatalf("unexpected clamp graph: %+v", output)
	}

	invalid := NewBuilder()
	invalid.Clamp(invalid.Input("input", dtype.F32, MustShape(1)), 2, -1)
	if invalid.Err() == nil {
		t.Fatal("clamp accepted reversed bounds")
	}
}

func TestBuilderQ8MoE(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(32, 1))
	router := builder.Input("router", dtype.F32, MustShape(32, 4))
	gate := builder.Input("gate", dtype.Q8_0, MustShape(32, 32, 4))
	up := builder.Input("up", dtype.Q8_0, MustShape(32, 32, 4))
	down := builder.Input("down", dtype.Q8_0, MustShape(32, 32, 4))
	output := builder.MoE(input, router, gate, up, down, 2, true, 1)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if output.Type != dtype.F32 || !output.Shape.Equal(input.Shape) {
		t.Fatalf("unexpected Q8_0 MoE output: %+v", output)
	}

	invalid := NewBuilder()
	invalidInput := invalid.Input("input", dtype.F32, MustShape(16, 1))
	invalidRouter := invalid.Input("router", dtype.F32, MustShape(16, 4))
	invalidGate := invalid.Input("gate", dtype.Q8_0, MustShape(16, 32, 4))
	invalidUp := invalid.Input("up", dtype.Q8_0, MustShape(16, 32, 4))
	invalidDown := invalid.Input("down", dtype.Q8_0, MustShape(32, 16, 4))
	invalid.MoE(invalidInput, invalidRouter, invalidGate, invalidUp, invalidDown, 2, true, 1)
	if invalid.Err() == nil {
		t.Fatal("Q8_0 MoE accepted an unaligned hidden width")
	}

	mixed := NewBuilder()
	mixedInput := mixed.Input("input", dtype.F32, MustShape(32, 1))
	mixedRouter := mixed.Input("router", dtype.F32, MustShape(32, 4))
	mixedGate := mixed.Input("gate", dtype.Q8_0, MustShape(32, 32, 4))
	mixedUp := mixed.Input("up", dtype.F32, MustShape(32, 32, 4))
	mixedDown := mixed.Input("down", dtype.Q8_0, MustShape(32, 32, 4))
	mixed.MoE(mixedInput, mixedRouter, mixedGate, mixedUp, mixedDown, 2, true, 1)
	if mixed.Err() == nil {
		t.Fatal("MoE accepted mixed expert storage types")
	}
}

func TestBuilderNativeQuantizedMoEFormats(t *testing.T) {
	for _, dataType := range []dtype.Type{
		dtype.Q4_0, dtype.Q4_1, dtype.Q5_0, dtype.Q5_1,
		dtype.Q8_0, dtype.Q8_1, dtype.Q2K, dtype.Q3K, dtype.Q4K, dtype.Q5K, dtype.Q6K, dtype.Q8K,
		dtype.IQ2XXS, dtype.IQ2XS, dtype.IQ2S, dtype.IQ3XXS, dtype.IQ3S, dtype.IQ1S, dtype.IQ1M,
		dtype.IQ4NL, dtype.IQ4XS, dtype.MXFP4, dtype.NVFP4,
		dtype.Q1_0, dtype.Q2_0, dtype.TQ1_0, dtype.TQ2_0,
	} {
		t.Run(dataType.String(), func(t *testing.T) {
			traits, ok := dataType.Traits()
			if !ok {
				t.Fatalf("missing traits for %s", dataType)
			}
			builder := NewBuilder()
			input := builder.Input("input", dtype.F32, MustShape(traits.BlockSize, 1))
			router := builder.Input("router", dtype.F32, MustShape(traits.BlockSize, 2))
			gate := builder.Input("gate", dataType, MustShape(traits.BlockSize, traits.BlockSize, 2))
			up := builder.Input("up", dataType, MustShape(traits.BlockSize, traits.BlockSize, 2))
			down := builder.Input("down", dataType, MustShape(traits.BlockSize, traits.BlockSize, 2))
			output := builder.MoE(input, router, gate, up, down, 1, false, 1)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			if output.Type != dtype.F32 {
				t.Fatalf("%s MoE output type = %s", dataType, output.Type)
			}
		})
	}
}

func TestBuilderGatedDeltaNet(t *testing.T) {
	builder := NewBuilder()
	q := builder.Input("q", dtype.F32, MustShape(4, 2, 3, 2))
	k := builder.Input("k", dtype.F32, MustShape(4, 1, 3, 2))
	v := builder.Input("v", dtype.F32, MustShape(4, 4, 3, 2))
	gate := builder.Input("gate", dtype.F32, MustShape(4, 4, 3, 2))
	beta := builder.Input("beta", dtype.F32, MustShape(1, 4, 3, 2))
	state := builder.Input("state", dtype.F32, MustShape(4, 4, 4, 2))
	output := builder.GatedDeltaNet(q, k, v, gate, beta, state)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(MustShape(16, 14)) {
		t.Fatalf("GatedDeltaNet output shape = %v", output.Shape.Slice())
	}
}

func TestBuilderQwen35LayoutOperations(t *testing.T) {
	builder := NewBuilder()
	projection := builder.Input("projection", dtype.F32, MustShape(16, 3))
	query := builder.GroupSlice(projection, 0, 2, 4, 4)
	gate := builder.GroupSlice(projection, 2, 2, 4, 4)
	transposed := builder.Transpose2D(projection)
	state := builder.Input("state", dtype.F32, MustShape(2, 16))
	convInput := builder.Concat(state, transposed, 0)
	packed := builder.Input("packed", dtype.F32, MustShape(8, 6))
	attention := builder.FlatSlice(packed, 0, 2, 4, 3)
	newState := builder.FlatSlice(packed, 24, 2, 2, 6)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !query.Shape.Equal(MustShape(2, 4, 3)) ||
		!gate.Shape.Equal(query.Shape) ||
		!convInput.Shape.Equal(MustShape(5, 16)) ||
		!attention.Shape.Equal(MustShape(2, 4, 3)) ||
		!newState.Shape.Equal(MustShape(2, 2, 6)) {
		t.Fatalf(
			"unexpected layout shapes: q=%v conv=%v attn=%v state=%v",
			query.Shape.Slice(),
			convInput.Shape.Slice(),
			attention.Shape.Slice(),
			newState.Shape.Slice(),
		)
	}
}

func TestBuilderGetRowsAndRoPE(t *testing.T) {
	builder := NewBuilder()
	table := builder.Input("table", dtype.F32, MustShape(8, 16))
	embedding := builder.GetRows(table, []uint32{3, 5})
	rope := builder.RoPENeoX(
		builder.Input("query", dtype.F32, MustShape(8, 2, 2)),
		[]uint32{4, 5},
		8,
		10000,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !embedding.Shape.Equal(MustShape(8, 2)) || !rope.Shape.Equal(MustShape(8, 2, 2)) {
		t.Fatalf("unexpected shapes %v and %v", embedding.Shape.Slice(), rope.Shape.Slice())
	}

	var positions [4][]uint32
	positions[0] = []uint32{1, 2}
	positions[1] = []uint32{3, 4}
	positions[2] = []uint32{5, 6}
	positions[3] = []uint32{7, 8}
	multi := builder.RoPEMulti(
		builder.Input("multi", dtype.F32, MustShape(8, 1, 2)),
		positions,
		[4]int32{1, 1, 1, 1},
		8,
		10000,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if multi.Op != OpRoPEMulti || !multi.Shape.Equal(MustShape(8, 1, 2)) {
		t.Fatalf("unexpected multi-axis RoPE tensor: %+v", multi)
	}
}

func TestBuilderQ8DeviceOperationsProduceF32(t *testing.T) {
	builder := NewBuilder()
	table := builder.Input("table", dtype.Q8_0, MustShape(32, 4))
	rows := builder.GetRows(table, []uint32{2})
	input := builder.Input("input", dtype.F32, MustShape(32, 2))
	product := builder.MulMat(table, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if rows.Type != dtype.F32 || product.Type != dtype.F32 {
		t.Fatalf("Q8 outputs have types %s and %s, want f32", rows.Type, product.Type)
	}
}

func TestBuilderQ6KDeviceOperationsProduceF32(t *testing.T) {
	builder := NewBuilder()
	table := builder.Input("table", dtype.Q6K, MustShape(256, 4))
	rows := builder.GetRows(table, []uint32{2})
	input := builder.Input("input", dtype.F32, MustShape(256, 2))
	product := builder.MulMat(table, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if rows.Type != dtype.F32 || product.Type != dtype.F32 {
		t.Fatalf("Q6_K outputs have types %s and %s, want f32", rows.Type, product.Type)
	}
}

func TestBuilderKQuantDeviceOperationsProduceF32(t *testing.T) {
	for _, dataType := range []dtype.Type{
		dtype.Q2K,
		dtype.Q3K,
		dtype.Q4K,
		dtype.Q5K,
		dtype.Q8K,
		dtype.IQ2XXS,
		dtype.IQ2XS,
		dtype.IQ2S,
		dtype.IQ3XXS,
		dtype.IQ3S,
		dtype.IQ1S,
		dtype.IQ1M,
		dtype.IQ4XS,
		dtype.TQ1_0,
		dtype.TQ2_0,
	} {
		builder := NewBuilder()
		table := builder.Input("table", dataType, MustShape(256, 4))
		rows := builder.GetRows(table, []uint32{2})
		input := builder.Input("input", dtype.F32, MustShape(256, 2))
		product := builder.MulMat(table, input)
		if err := builder.Err(); err != nil {
			t.Fatalf("%s: %v", dataType, err)
		}
		if rows.Type != dtype.F32 || product.Type != dtype.F32 {
			t.Fatalf("%s outputs have types %s and %s, want f32", dataType, rows.Type, product.Type)
		}
	}
}

func TestBuilderClassicQuantDeviceOperationsProduceF32(t *testing.T) {
	for _, dataType := range []dtype.Type{
		dtype.Q4_0,
		dtype.Q4_1,
		dtype.Q5_0,
		dtype.Q5_1,
		dtype.Q8_1,
		dtype.IQ4NL,
		dtype.MXFP4,
	} {
		builder := NewBuilder()
		table := builder.Input("table", dataType, MustShape(32, 4))
		rows := builder.GetRows(table, []uint32{2})
		input := builder.Input("input", dtype.F32, MustShape(32, 2))
		product := builder.MulMat(table, input)
		if err := builder.Err(); err != nil {
			t.Fatalf("%s: %v", dataType, err)
		}
		if rows.Type != dtype.F32 || product.Type != dtype.F32 {
			t.Fatalf("%s outputs have types %s and %s, want f32", dataType, rows.Type, product.Type)
		}
	}
}

func TestBuilderSmallQuantDeviceOperationsProduceF32(t *testing.T) {
	for _, test := range []struct {
		dataType dtype.Type
		width    uint64
	}{
		{dtype.Q1_0, 128},
		{dtype.Q2_0, 64},
		{dtype.NVFP4, 64},
	} {
		builder := NewBuilder()
		table := builder.Input("table", test.dataType, MustShape(test.width, 4))
		rows := builder.GetRows(table, []uint32{2})
		input := builder.Input("input", dtype.F32, MustShape(test.width, 2))
		product := builder.MulMat(table, input)
		if err := builder.Err(); err != nil {
			t.Fatalf("%s: %v", test.dataType, err)
		}
		if rows.Type != dtype.F32 || product.Type != dtype.F32 {
			t.Fatalf("%s outputs have types %s and %s, want f32", test.dataType, rows.Type, product.Type)
		}
	}
}

func TestTopologicalRejectsCycle(t *testing.T) {
	node := &Tensor{ID: 1, Type: dtype.F32, Shape: MustShape(1), Op: OpScale}
	node.Inputs = []*Tensor{node}
	if _, err := Topological(node); err == nil {
		t.Fatal("cyclic graph was accepted")
	}
}

func TestBuilderGroupedMulMatShape(t *testing.T) {
	builder := NewBuilder()
	left := builder.Input("left", dtype.F32, MustShape(3, 4, 2))
	right := builder.Input("right", dtype.F32, MustShape(3, 2, 5))
	output := builder.GroupedMulMat(left, right)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(MustShape(4, 2, 5)) || output.Op != OpGroupedMulMat {
		t.Fatalf("unexpected grouped matmul output: %+v", output)
	}
}
