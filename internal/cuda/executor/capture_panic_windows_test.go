//go:build windows

package executor

import (
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestAPanicWhileCapturingEndsTheCapture makes a launcher panic while its execution is captured into a graph
// (executor-close/capture-panic), as an unchecked assertion in a launcher would. The worker turns the panic into an
// error and the executor goes on serving, so the stream must be out of capture: the next execution, with the launcher
// sound again, captures afresh and gives the graph's values.
func TestAPanicWhileCapturingEndsTheCapture(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8)
	input := builder.Input("input", dtype.F32, shape)
	output := builder.Scale(input, 2)
	compiled := testcheck.Must(Compile(output))(t)
	if len(compiled.launches) == 0 {
		t.Fatal("the graph launches no kernel")
	}
	cuda := newFixtureExecutor(t)
	feeds := map[*tensor.Tensor]reference.Value{input: {Shape: shape, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}}}
	launched := compiled.launches[0]
	sound := compiled.nodes[launched].launcher
	compiled.nodes[launched].launcher = func(*device.State, functionSet, *blasState, *q8InputState, *tensor.Tensor, tensor.Attributes, launchPointerFrame) error {
		panic("a launcher's fault while its stream captures")
	}
	if _, err := cuda.ExecuteRetainedCompiled(ctx, compiled, feeds, nil, nil, nil); err == nil {
		t.Fatal("an execution whose launcher panicked succeeded")
	}
	compiled.nodes[launched].launcher = sound
	retained := testcheck.Must(cuda.ExecuteRetainedCompiled(ctx, compiled, feeds, nil, nil, nil))(t)
	got := testcheck.Must(retained.CopyToHost(ctx, output))(t)
	testcheck.NoError(t, retained.Release(ctx))
	if want := []float32{2, 4, 6, 8, 10, 12, 14, 16}; !slices.Equal(got.Data, want) {
		t.Fatalf("after the panic the graph gives %v, want %v", got.Data, want)
	}
}
