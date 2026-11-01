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

func TestGatedDeltaChunkLifetime(t *testing.T) {
	cudatest.Require(t)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	before, err := worker.MemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for _, tokens := range []uint64{65, 129, 1, 65} {
		out := chunkTestGraph(t, tokens)
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
			t.Fatal("phase release retains chunk workspace")
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
		t.Fatalf("close leaks owned device memory: before=%+v after=%+v error=%v", before, after, err)
	}
}
