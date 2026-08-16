package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestReserveExactWeightStagingRestoresFullFootprint(t *testing.T) {
	const inner, rows, tokens = 4096, 4096, 8
	builder := tensor.NewBuilder()
	weight := builder.Input("weight", dtype.BF16, tensor.MustShape(inner, rows))
	right := builder.Input("right", dtype.F32, tensor.MustShape(inner, tokens))
	compiled, err := Compile(builder.MulMat(weight, right))
	if err != nil {
		t.Fatal(err)
	}
	full := uint64(inner) * uint64(rows) * 4
	if full <= nativeWeightStagingLimitBytes {
		t.Fatalf("fixture footprint %d does not exceed the bounded limit %d", full, nativeWeightStagingLimitBytes)
	}
	if compiled.matmulStagingBytes != nativeWeightStagingLimitBytes {
		t.Fatalf("bounded staging = %d, want %d", compiled.matmulStagingBytes, nativeWeightStagingLimitBytes)
	}
	if err := compiled.ReserveExactWeightStaging(); err != nil {
		t.Fatal(err)
	}
	if compiled.matmulStagingBytes != full {
		t.Fatalf("exact staging = %d, want %d", compiled.matmulStagingBytes, full)
	}
}

func TestReserveExactWeightStagingLeavesDecodeAndTensorCoreAlone(t *testing.T) {
	const inner, rows = 4096, 65536
	decodeBuilder := tensor.NewBuilder()
	decodeWeight := decodeBuilder.Input("weight", dtype.BF16, tensor.MustShape(inner, rows))
	decodeRight := decodeBuilder.Input("right", dtype.F32, tensor.MustShape(inner, 1))
	decodeCompiled, err := Compile(decodeBuilder.MulMat(decodeWeight, decodeRight))
	if err != nil {
		t.Fatal(err)
	}
	tcBuilder := tensor.NewBuilder()
	tcBuilder.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
	tcWeight := tcBuilder.Input("weight", dtype.BF16, tensor.MustShape(inner, rows))
	tcRight := tcBuilder.Input("right", dtype.F32, tensor.MustShape(inner, 8))
	tcCompiled, err := Compile(tcBuilder.MulMat(tcWeight, tcRight))
	if err != nil {
		t.Fatal(err)
	}
	for name, compiled := range map[string]*CompiledGraph{"decode": decodeCompiled, "tensor-core": tcCompiled} {
		before := compiled.matmulStagingBytes
		if err := compiled.ReserveExactWeightStaging(); err != nil {
			t.Fatal(err)
		}
		if compiled.matmulStagingBytes != before {
			t.Fatalf("%s staging changed %d -> %d", name, before, compiled.matmulStagingBytes)
		}
	}
	if err := (*CompiledGraph)(nil).ReserveExactWeightStaging(); err == nil {
		t.Fatal("nil compiled graph accepted")
	}
}
