package executor

import (
	"math"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"testing"
)

func TestNativeL2Selection(t *testing.T) {
	for _, native := range []bool{false, true} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(128, 22))
		out := b.L2Norm(x, 1e-6)
		if native {
			out = b.L2NormNativeBF16(x, 1e-6)
		}
		graph, err := Compile(out)
		if err != nil {
			t.Fatal(err)
		}
		next := out.Attrs.(tensor.L2NormAttributes)
		next.NativeBF16 = !native
		if err = graph.NewRuntimeAttributes().Set(out, next); err == nil {
			t.Fatal("runtime changed native policy")
		}
	}
	for _, width := range []uint64{1, 127, 129, 262144} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(width, 22))
		if b.L2NormNativeBF16(x, 1e-6) != nil || b.Err() == nil {
			t.Fatalf("unsupported width %d admitted", width)
		}
	}
	for _, epsilon := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(128))
		if b.L2NormNativeBF16(x, epsilon) != nil || b.Err() == nil {
			t.Fatal("unsupported epsilon", epsilon)
		}
	}
	b := tensor.NewBuilder()
	if b.L2NormNativeBF16(b.Input("bf16", dtype.BF16, tensor.MustShape(128)), 1e-6) != nil || b.Err() == nil {
		t.Fatal("native L2 accepted non-F32 storage")
	}
	if _, _, err := tensor.NativeReductionGeometry(tensor.MustShape(128, 1<<32), true); err == nil {
		t.Fatal("overflow rows admitted")
	}
}
