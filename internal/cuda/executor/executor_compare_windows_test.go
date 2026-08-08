//go:build windows

package executor

import (
	"context"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

const cudaFixtureDevice = 0

const (
	accuracyExact       = 0
	accuracyFP32Tight   = 1e-6
	accuracyFP32        = 5e-6
	accuracyProjection  = 1e-5
	accuracySparse      = 2e-4
	accuracyMLA         = 3e-4
	accuracyState       = 5e-5
	accuracyStateLoose  = 7e-5
	accuracyQuantKernel = 1e-4
	accuracyModel       = 5e-4
	accuracyAttention   = 6e-4
	accuracyRecurrent   = 8e-4
	accuracyModelLoose  = 2e-3
	accuracyElementwise = 2e-5
	accuracyQuantized   = 1e-2
)

type graphOutputCheck struct {
	output    *tensor.Tensor
	tolerance float64
}

func newFixtureExecutor(t testing.TB) *Executor {
	t.Helper()
	cudatest.Require(t)
	executor, err := New(cudaFixtureDevice)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = executor.Close() })
	return executor
}

func newFixtureWorker(t testing.TB) *device.Worker {
	t.Helper()
	cudatest.Require(t)
	worker, err := device.New(cudaFixtureDevice)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = worker.Close() })
	return worker
}

func newFixtureExecutorWithWorker(t testing.TB, worker *device.Worker) *Executor {
	t.Helper()
	executor, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = executor.Close() })
	return executor
}

func fixturePositions(tokens uint32) []uint32 {
	positions := make([]uint32, tokens)
	for index := range positions {
		positions[index] = uint32(index)
	}
	return positions
}

func fixtureShapeBytes(t testing.TB, shape tensor.Shape, dataType dtype.Type) uint64 {
	t.Helper()
	bytes, err := shape.Bytes(dataType)
	if err != nil {
		t.Fatal(err)
	}
	return bytes
}

func checkCUDAGraph(
	t *testing.T,
	feeds map[*tensor.Tensor]reference.Value,
	checks ...graphOutputCheck,
) {
	t.Helper()
	cudatest.Require(t)
	outputs := make([]*tensor.Tensor, len(checks))
	for index, check := range checks {
		outputs[index] = check.output
	}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for _, check := range checks {
		compare(t, got[check.output].Data, want[check.output].Data, check.tolerance)
	}
}

func uniformGraphChecks(outputs []*tensor.Tensor, tolerance float64) []graphOutputCheck {
	checks := make([]graphOutputCheck, len(outputs))
	for index, output := range outputs {
		checks[index] = graphOutputCheck{output: output, tolerance: tolerance}
	}
	return checks
}
