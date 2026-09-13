//go:build windows

package executor

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestWeightStagingPhaseRelease(t *testing.T) {
	cudatest.Require(t)
	// Assertions use owned allocations and exact outputs; shared admission suffices.
	builder := tensor.NewBuilder()
	weight := builder.Input("weight", dtype.BF16, tensor.MustShape(64, 64))
	right := builder.Input("right", dtype.F32, tensor.MustShape(64, 8))
	output := builder.MulMat(weight, right)
	graph, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	if err := graph.ReserveExactWeightStaging(); err != nil {
		t.Fatal(err)
	}
	weights := patternedValue(weight.Shape, 7, 0.25, 0)
	storage, err := quant.Quantize(dtype.BF16, weights.Data)
	if err != nil {
		t.Fatal(err)
	}
	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)
	inputs := graph.NewDeviceInputs()
	if err := inputs.Set(weight, pointer); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{right: patternedValue(right.Shape, 5, 0.5, 0)}
	first, err := cuda.ExecuteRetainedCompiled(t.Context(), graph, feeds, inputs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := first.CopyToHost(t.Context(), output)
	if err != nil {
		t.Fatal(err)
	}
	var staging, before uint64
	if err := cuda.worker.Do(t.Context(), func(state *device.State) error {
		staging = cuda.resources.blas.stagingBytes
		before = state.Driver.MemoryStats().CurrentBytes
		return nil
	}); err != nil || staging == 0 {
		t.Fatalf("staging acquisition: bytes=%d err=%v", staging, err)
	}
	canceled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if err := cuda.ReleaseWeightStaging(canceled); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled phase release", err)
	}
	if err := cuda.ReleaseWeightStaging(context.WithoutCancel(canceled)); err != nil {
		t.Fatal(err)
	}
	if err := cuda.worker.Do(t.Context(), func(state *device.State) error {
		if got := state.Driver.MemoryStats().CurrentBytes; got != before-staging {
			t.Errorf("phase release retained bytes=%d want=%d", got, before-staging)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	metrics, err := cuda.Metrics(t.Context())
	if err != nil || metrics.GraphCacheEntries != 0 {
		t.Fatal("phase release kept captured graph references", metrics, err)
	}
	got, err := first.CopyToHost(t.Context(), output)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got.Data, want.Data, accuracyExact)
	if err := first.Release(t.Context()); err != nil {
		t.Fatal(err)
	}
	second, err := cuda.ExecuteRetainedCompiled(t.Context(), graph, feeds, inputs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err = second.CopyToHost(t.Context(), output)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got.Data, want.Data, accuracyExact)
	if err := second.Release(t.Context()); err != nil {
		t.Fatal(err)
	}
	t.Logf("released staging bytes=%d; retained output and recaptured execution are exact", staging)
}
