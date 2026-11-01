//go:build windows

package executor

import (
	"context"
	"errors"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestNativeRotaryLifetime(t *testing.T) {
	cudatest.Require(t)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	before, err := worker.MemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for _, tokens := range []uint64{22, 151, 1, 22} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(256, 4, tokens))
		positions := make([]uint32, tokens)
		for i := range positions {
			positions[i] = uint32(i)
		}
		out := b.RoPEWithOptions(b.BF16Round(x), tensor.RoPEOptions{NativeBF16: true, Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: 64, FrequencyBase: 10000000, FrequencyScale: 1})
		feeds := map[*tensor.Tensor]reference.Value{x: patternedValue(x.Shape, 7, .125, -.5)}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := cuda.Execute(ctx, []*tensor.Tensor{out}, feeds); !errors.Is(err, context.Canceled) {
			t.Fatal("canceled execution", err)
		}
		first, err := cuda.Execute(t.Context(), []*tensor.Tensor{out}, feeds)
		if err != nil {
			t.Fatal(err)
		}
		if err := cuda.ReleaseWeightStaging(t.Context()); err != nil {
			t.Fatal(err)
		}
		again, err := cuda.Execute(t.Context(), []*tensor.Tensor{out}, feeds)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, again[out].Data, first[out].Data, accuracyExact)
	}
	if err := cuda.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := worker.MemoryStats(t.Context())
	if err != nil || after.CurrentBytes != before.CurrentBytes {
		t.Fatalf("native rotary release: before=%+v after=%+v error=%v", before, after, err)
	}
}
