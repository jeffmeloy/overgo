//go:build windows

package executor

import (
	"context"
	"errors"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestNativeAttentionLifetime(t *testing.T) {
	cudatest.Require(t)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	before, err := worker.MemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for _, tokens := range []uint64{22, 151, 1, 22} {
		out := nativeAttentionTestGraph(t, tokens)
		feeds := map[*tensor.Tensor]reference.Value{}
		for i, in := range out.Inputs {
			feeds[in] = patternedValue(in.Shape, i+3, .01, -.1)
		}
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
		if cuda.resources.blas.stagingBytes != 0 {
			t.Fatal("phase release retains native attention workspace")
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
		t.Fatalf("native attention resource release: before=%+v after=%+v error=%v", before, after, err)
	}
}
