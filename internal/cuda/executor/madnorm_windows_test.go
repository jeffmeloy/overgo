//go:build windows

package executor

import (
	"context"
	"math"
	"math/rand"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestMADNormGraphDeviceMatchesReference(t *testing.T) {
	cudatest.Require(t)
	const rows, width = 7, 8
	shape := tensor.MustShape(width, rows)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, shape)
	output := builder.MADNorm(input, 1e-8)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	values := make([]float32, rows*width)
	for index := range values {
		values[index] = rng.Float32()*2 - 1
	}
	value, err := reference.NewValue(shape, values)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: value}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.WithoutCancel(t.Context()), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for index := range value.Data {
		worst = max(worst, math.Abs(float64(got[output].Data[index]-want[output].Data[index])))
	}
	if worst > 1e-5 {
		t.Fatalf("MADNorm graph CUDA/reference worst=%.3e", worst)
	}
	t.Logf("MADNorm graph CUDA/reference worst=%.3e", worst)
}
