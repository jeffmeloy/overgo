package executor

import (
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
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
