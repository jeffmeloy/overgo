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

func TestNativeMathLifetime(t *testing.T) {
	cudatest.Require(t)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	before, err := worker.MemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for _, rows := range []uint64{1, 22, 151} {
		b := tensor.NewBuilder()
		x := b.Input("x", dtype.F32, tensor.MustShape(4096, rows))
		out := b.RMSNormNative(b.SoftplusNative(b.ExpNative(x)), 1e-6)
		out = b.L2NormNativeBF16(b.BF16Round(out), 1e-6)
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
		again, err := cuda.Execute(t.Context(), []*tensor.Tensor{out}, feeds)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, again[out].Data, first[out].Data, accuracyExact)
		if len(cuda.resources.modules) != len(kernelModuleAssets) {
			t.Fatal("module ownership differs from manifest")
		}
	}
	if err := cuda.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := worker.MemoryStats(t.Context())
	if err != nil || after.CurrentBytes != before.CurrentBytes || len(cuda.resources.modules) != 0 {
		t.Fatalf("native resource release: before=%+v after=%+v error=%v", before, after, err)
	}
}
