package executor

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

const benchmarkGraphLayers = 32

func benchmarkGraph() *tensor.Tensor {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(256, 4)
	current := builder.Input("input", dtype.F32, shape)
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(256, 256))
	for range benchmarkGraphLayers {
		residual := current
		current = builder.MulMat(weight, current)
		current = builder.SiLU(current)
		current = builder.Add(current, residual)
	}
	return current
}

func BenchmarkCompileGraph(b *testing.B) {
	output := benchmarkGraph()
	if _, err := Compile(output); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := Compile(output); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLaunchPointerResolution(b *testing.B) {
	output := benchmarkGraph()
	compiled, err := Compile(output)
	if err != nil {
		b.Fatal(err)
	}
	values := make([]driver.DevicePtr, len(compiled.order))
	for index := range values {
		values[index] = driver.DevicePtr(index + 1)
	}
	node := compiled.order[len(compiled.order)/2]
	slot := compiled.orderIndexes[node]
	frame := launchPointerFrame{values: values, slots: []int{slot, slot}}
	b.Run("map", func(b *testing.B) {
		pointers := graphPointerTable{indexes: compiled.orderIndexes, values: values}
		for range b.N {
			if pointers.get(node) == 0 {
				b.Fatal("missing pointer")
			}
		}
	})
	b.Run("slot", func(b *testing.B) {
		for range b.N {
			if frame.input(0) == 0 {
				b.Fatal("missing pointer")
			}
		}
	})
}
