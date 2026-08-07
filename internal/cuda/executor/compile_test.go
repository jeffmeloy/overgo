package executor

import (
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
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
