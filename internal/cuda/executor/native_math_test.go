package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestNativeMathSelection(t *testing.T) {
	for _, tc := range []struct {
		width, rows       uint64
		lanes, reductions uint32
	}{
		{128, 1, 32, 1}, {4096, 1, 512, 1}, {4096, 10, 64, 1}, {4096, 22, 32, 1}, {8192, 22, 32, 16}, {32768, 22, 32, 16},
	} {
		x, y, err := tensor.NativeReductionGeometry(tensor.MustShape(tc.width, tc.rows), false)
		if err != nil || x != tc.lanes || y != tc.reductions {
			t.Fatalf("geometry %+v: %d/%d %v", tc, x, y, err)
		}
	}
	for _, width := range []uint64{1, 129, 262144} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(width, 22))
		if _, err := Compile(b.RMSNormNative(x, 1e-6)); err == nil {
			t.Fatalf("unsupported native width %d admitted", width)
		}
	}
	for _, mode := range []string{"default", "native", "native-mean"} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(4096, 10))
		w := b.Input("w", dtype.F32, tensor.MustShape(4096))
		norm := b.RMSNorm(x, 1e-6)
		if mode == "native" {
			norm = b.RMSNormNative(x, 1e-6)
		} else if mode == "native-mean" {
			norm = b.RMSNormNativeMean(x, 1e-6)
		}
		out := b.Multiply(norm, w)
		graph, err := Compile(out)
		if err != nil {
			t.Fatal(err)
		}
		fusion := graph.nodes[graph.orderIndexes[out]].fusion
		if fused := fusion != nil && fusion.kind == compiledFusionWeightedRMS; fused != (mode == "default") {
			t.Fatalf("normalization fusion violates arithmetic selection: mode=%s", mode)
		}
	}
}

func TestNativeSiLUSelection(t *testing.T) {
	for _, native := range []bool{false, true} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(16))
		y := b.Input("y", dtype.F32, x.Shape)
		activation := b.SiLU(x)
		if native {
			activation = b.SiLUNative(x)
		}
		out := b.Multiply(activation, y)
		graph, err := Compile(out)
		if err != nil {
			t.Fatal(err)
		}
		fusion := graph.nodes[graph.orderIndexes[out]].fusion
		if fused := fusion != nil && fusion.kind == compiledFusionActivatedGate; fused == native {
			t.Fatalf("activation fusion violates source arithmetic: native=%v", native)
		}
	}
}
