package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestCompilePinsTopologyAndMemoryPlan(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4, 2)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.Add(left, right)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.order) != 3 || len(compiled.outputs) != 1 || compiled.outputs[0] != output {
		t.Fatalf("compiled graph = %+v", compiled)
	}
	if compiled.memory.ArenaSize == 0 || compiled.needBlas {
		t.Fatalf("compiled memory/BLAS = %d/%t", compiled.memory.ArenaSize, compiled.needBlas)
	}
}

func TestCompileDetectsBLASAndRejectsNilGraph(t *testing.T) {
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.F32, tensor.MustShape(2, 2))
	right := builder.Input("right", dtype.F32, tensor.MustShape(2, 2))
	compiled, err := Compile(builder.MulMat(left, right))
	if err != nil {
		t.Fatal(err)
	}
	if !compiled.needBlas {
		t.Fatal("F32 matrix graph did not request BLAS")
	}
	if _, err := Compile(nil); err == nil {
		t.Fatal("nil graph compiled")
	}
}

func TestDeviceBufferBucket(t *testing.T) {
	for _, fixture := range []struct {
		size uint64
		want uint64
	}{
		{1, minimumDeviceBufferBytes},
		{minimumDeviceBufferBytes, minimumDeviceBufferBytes},
		{minimumDeviceBufferBytes + 1, minimumDeviceBufferBytes * 2},
		{minimumDeviceBufferBytes * 4, minimumDeviceBufferBytes * 4},
	} {
		got, err := deviceBufferBucket(fixture.size)
		if err != nil || got != fixture.want {
			t.Fatalf("bucket(%d) = (%d, %v), want (%d, nil)", fixture.size, got, err, fixture.want)
		}
	}
	if _, err := deviceBufferBucket(0); err == nil {
		t.Fatal("expected zero-size buffer rejection")
	}
}

func TestRetainedOutputLayoutUsesOneAlignedSpan(t *testing.T) {
	const fixtureWidth = 5
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(fixtureWidth, 1))
	first := builder.SiLU(input)
	second := builder.Sigmoid(first)
	order, err := tensor.Topological(second)
	if err != nil {
		t.Fatal(err)
	}
	offsets, bytes, err := retainedOutputLayout(order, map[*tensor.Tensor]struct{}{
		first: {}, second: {},
	})
	if err != nil {
		t.Fatal(err)
	}
	if offsets[first]%graphArenaAlignment != 0 || offsets[second]%graphArenaAlignment != 0 {
		t.Fatalf("retained offsets are not aligned: %v", offsets)
	}
	if offsets[first] == offsets[second] || bytes <= offsets[second] {
		t.Fatalf("retained layout overlaps or truncates: offsets %v bytes %d", offsets, bytes)
	}
}

func TestCompileFusesSingleUseWeightedRMSNorm(t *testing.T) {
	const fixtureWidth = 8
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(fixtureWidth, 2))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(fixtureWidth))
	normalized := builder.RMSNorm(input, 1e-5)
	output := builder.Multiply(normalized, weight)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	fusion, ok := compiled.weightedRMS[output]
	if !ok || fusion.normalization != normalized || fusion.weight != weight {
		t.Fatalf("weighted RMSNorm fusion = %+v, available %t", fusion, ok)
	}
	if _, skipped := compiled.skipped[normalized]; !skipped {
		t.Fatal("fused RMSNorm launch was not skipped")
	}
}

func TestCompileFusesResidualAddIntoWeightedRMSNorm(t *testing.T) {
	const fixtureWidth = 8
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(fixtureWidth, 2)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(fixtureWidth))
	residual := builder.Add(left, right)
	normalized := builder.RMSNorm(residual, 1e-5)
	output := builder.Multiply(normalized, weight)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	fusion, ok := compiled.weightedRMS[output]
	if !ok || fusion.addLeft != left || fusion.addRight != right {
		t.Fatalf("weighted residual RMSNorm fusion = %+v, available %t", fusion, ok)
	}
	if _, skipped := compiled.skipped[residual]; !skipped {
		t.Fatal("fused residual launch was not skipped")
	}
}

func TestCompileFusesWeightedRMSNormAndActivatedGate(t *testing.T) {
	const fixtureWidth = 8
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(fixtureWidth, 2)
	input := builder.Input("input", dtype.F32, shape)
	gate := builder.Input("gate", dtype.F32, shape)
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(fixtureWidth))
	normalized := builder.RMSNorm(input, 1e-5)
	weighted := builder.Multiply(normalized, weight)
	activation := builder.SiLU(gate)
	output := builder.Multiply(weighted, activation)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	fusion, ok := compiled.weightedRMSGate[output]
	if !ok || fusion.normalization != normalized || fusion.weight != weight ||
		fusion.gate != gate || fusion.kind != activatedGateSiLU {
		t.Fatalf("weighted RMS gate fusion = %+v, available %t", fusion, ok)
	}
	for _, skipped := range []*tensor.Tensor{normalized, weighted, activation} {
		if _, ok := compiled.skipped[skipped]; !ok {
			t.Fatalf("fused tensor %d was not skipped", skipped.ID)
		}
	}
	if _, exists := compiled.weightedRMS[weighted]; exists {
		t.Fatal("subsumed weighted RMS fusion remains active")
	}
}
