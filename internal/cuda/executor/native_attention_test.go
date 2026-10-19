package executor

import (
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func nativeAttentionTestGraph(t *testing.T, tokens uint64) *tensor.Tensor {
	t.Helper()
	b := tensor.NewBuilder()
	q := b.Input("q", dtype.F32, tensor.MustShape(8, 4, tokens, 2))
	k := b.Input("k", dtype.F32, tensor.MustShape(8, 2, tokens, 2))
	v := b.Input("v", dtype.F32, tensor.MustShape(8, 2, tokens, 2))
	out := b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: .25, Causal: true, NativeMath: true})
	if b.Err() != nil {
		t.Fatal(b.Err())
	}
	return out
}

func TestNativeAttentionSelection(t *testing.T) {
	for _, native := range []bool{false, true} {
		b := tensor.NewBuilder()
		q := b.BF16Round(b.Input("q", dtype.F32, tensor.MustShape(32, 4, 16)))
		k := b.BF16Round(b.Input("k", dtype.F32, tensor.MustShape(32, 2, 16)))
		v := b.BF16Round(b.Input("v", dtype.F32, tensor.MustShape(32, 2, 16)))
		out := b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: .25, NativeMath: native})
		compiled, err := Compile(out)
		if err != nil {
			t.Fatal(err)
		}
		fusion := compiled.nodes[compiled.orderIndexes[out]].fusion
		if fused := fusion != nil && fusion.kind == compiledFusionBF16Attention; fused == native {
			t.Fatalf("attention fusion violates source selection: native=%v", native)
		}
	}
	native := nativeAttentionTestGraph(t, 22)
	policy := native.Attrs.(tensor.AttentionAttributes)
	policy.Causal = false
	native.Attrs = policy
	compiled, err := Compile(native)
	if err != nil {
		t.Fatal(err)
	}
	parameters := compiled.NewRuntimeAttributes()
	if err := parameters.Set(native, policy); err != nil {
		t.Fatal(err)
	}
	policy.KeyValueTokens = 10
	if err := parameters.Set(native, policy); err == nil {
		t.Fatal("native static key extent changed at runtime")
	}
	for _, tokens := range []uint64{1, 10, 22, 151, 2048, 2049, 4096} {
		node := nativeAttentionTestGraph(t, tokens)
		graph, err := Compile(node)
		if err != nil {
			t.Fatal(err)
		}
		// Four token matrices plus one score matrix, eight head batches, F32.
		want := (4*8*tokens + tokens*tokens) * 8 * 4
		if !graph.needBlas || graph.matmulStagingBytes != want || graph.attentionScoreBytes != 0 {
			t.Fatalf("native staging differs: got=%d want=%d", graph.matmulStagingBytes, want)
		}
	}
	for _, options := range []tensor.AttentionOptions{
		{Scale: 1, NativeMath: true, NaiveF32: true}, {Scale: 1, NativeMath: true, MemoryEfficientF32: true},
		{Scale: 1, NativeMath: true, Causal: true, Window: 1}, {Scale: 1, NativeMath: true, Softcap: 1},
		{Scale: float32(math.NaN()), NativeMath: true}, {Scale: float32(math.Inf(1)), NativeMath: true},
	} {
		node := nativeAttentionTestGraph(t, 10)
		b := tensor.NewBuilder()
		in := node.Inputs
		if b.AttentionWithOptions(in[0], in[1], in[2], options) != nil || b.Err() == nil {
			t.Fatal("conflicting native policy admitted", options)
		}
	}
	for _, invalid := range []string{"keys", "width", "groups", "elements", "capacity"} {
		t.Run(invalid, func(t *testing.T) {
			node := nativeAttentionTestGraph(t, 22)
			switch invalid {
			case "keys":
				node.Inputs[1].Shape.Dims[2] = tensor.NativeAttentionSoftmaxWidth + 1
			case "width":
				node.Inputs[0].Shape.Dims[0] = math.MaxInt32 + 1
			case "groups":
				node.Inputs[0].Shape.Dims[1] = math.MaxInt32
			case "elements":
				node.Inputs[0].Shape.Dims[2] = math.MaxInt32
			case "capacity":
				a := node.Attrs.(tensor.AttentionAttributes)
				a.KeyValueTokens = 10
				node.Attrs = a
			}
			if _, err := nativeAttentionGeometry(node); err == nil {
				t.Fatal("unsupported native geometry admitted")
			}
		})
	}
	b := tensor.NewBuilder()
	x := b.Input("x", dtype.F32, tensor.MustShape(8, 2, tensor.NativeAttentionSoftmaxWidth+1))
	if b.AttentionWithOptions(x, x, x, tensor.AttentionOptions{Scale: 1, NativeMath: true}) != nil || b.Err() == nil {
		t.Fatal("native softmax limit ignored")
	}
	b = tensor.NewBuilder()
	out := b.AttentionWithOptions(x, x, x, tensor.AttentionOptions{Scale: 1})
	if b.Err() != nil || out.Attrs.(tensor.AttentionAttributes).NativeMath {
		t.Fatal("default attention changed", b.Err())
	}
}
