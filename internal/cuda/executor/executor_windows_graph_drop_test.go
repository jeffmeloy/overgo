package executor

import (
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// A replayed exec carries kernel parameters by value, so once device memory
// a captured graph references is freed the exec must go: after two retained
// executions the second replays, after the drop the next captures again, and
// a compile after the drop matches only through its own serial.
func TestExecutorDropsGraphExecsBeforeFreedMemoryReplays(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8)
	input := builder.Input("input", dtype.F32, shape)
	output := builder.Scale(input, 2)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	feeds := map[*tensor.Tensor]reference.Value{
		input: {Shape: shape, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
	}
	execute := func() {
		t.Helper()
		retained, executeErr := cuda.ExecuteRetainedCompiled(ctx, compiled, feeds, nil, nil, nil)
		if executeErr != nil {
			t.Fatal(executeErr)
		}
		if releaseErr := retained.Release(ctx); releaseErr != nil {
			t.Fatal(releaseErr)
		}
	}
	execute()
	execute()
	before, err := cuda.Metrics(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if before.GraphCaptures != 1 || before.GraphCacheHits != 1 || before.GraphCacheEntries != 1 {
		t.Fatalf("steady replay metrics = %+v", before)
	}
	if err := cuda.worker.Do(ctx, func(state *device.State) error {
		return cuda.DropGraphExecs(state)
	}); err != nil {
		t.Fatal(err)
	}
	dropped, err := cuda.Metrics(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if dropped.GraphDrops != 1 || dropped.GraphCacheEntries != 0 {
		t.Fatalf("drop metrics = %+v", dropped)
	}
	execute()
	after, err := cuda.Metrics(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if after.GraphCaptures != 2 || after.GraphCacheHits != 1 || after.GraphCacheEntries != 1 {
		t.Fatalf("post-drop metrics = %+v", after)
	}
	recompiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	if recompiled.serial == compiled.serial || recompiled.serial == 0 {
		t.Fatalf("compiled serials %d and %d are not distinct", compiled.serial, recompiled.serial)
	}
}
