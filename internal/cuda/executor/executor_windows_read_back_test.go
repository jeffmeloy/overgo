//go:build windows

package executor

import (
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestRetainedRunReadsBackItsNamedOutputs holds a retained run of a graph with a read-back output to the values the
// card computed, returned by CopyToHost with no copy of its own, captured and replayed alike; an output it does not
// name still comes down through a copy, and an output of another graph cannot be named.
func TestRetainedRunReadsBackItsNamedOutputs(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8)
	input := builder.Input("input", dtype.F32, shape)
	doubled, tripled := builder.Scale(input, 2), builder.Scale(input, 3)
	compiled, err := Compile(doubled, tripled)
	if err != nil {
		t.Fatal(err)
	}
	if err := compiled.ReadBack(doubled); err != nil {
		t.Fatal(err)
	}
	if err := compiled.ReadBack(builder.Scale(input, 4)); err == nil {
		t.Fatal("an output of another graph was named for read-back")
	}
	if err := compiled.ReadBack(doubled); err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	feeds := map[*tensor.Tensor]reference.Value{input: {Shape: shape, Data: data}}
	copies := func() uint64 {
		t.Helper()
		var stats driver.ExecutionStats
		if err := cuda.worker.Do(ctx, func(state *device.State) error { stats = state.Driver.ExecutionStats(); return nil }); err != nil {
			t.Fatal(err)
		}
		return stats.DeviceToHostCopies
	}
	for run := range 2 {
		retained, err := cuda.ExecuteRetainedCompiled(ctx, compiled, feeds, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		before := copies()
		read, readErr := retained.CopyToHost(ctx, doubled)
		if after := copies(); readErr != nil || after != before {
			t.Fatalf("run %d: the read-back output took %d copies after the run (%v), want none", run, after-before, readErr)
		}
		other, otherErr := retained.CopyToHost(ctx, tripled)
		if after := copies(); otherErr != nil || after == before {
			t.Fatalf("run %d: the output not read back came down with no copy (%v)", run, otherErr)
		}
		if err := retained.Release(ctx); err != nil {
			t.Fatal(err)
		}
		for index, value := range data {
			if read.Data[index] != 2*value || other.Data[index] != 3*value {
				t.Fatalf("run %d: read %v and %v, want twice and three times %v", run, read.Data, other.Data, data)
			}
		}
		if !slices.Equal(read.Shape.Dims[:read.Shape.Rank], shape.Dims[:shape.Rank]) {
			t.Fatalf("run %d: read shape %v, want %v", run, read.Shape, shape)
		}
	}
}
