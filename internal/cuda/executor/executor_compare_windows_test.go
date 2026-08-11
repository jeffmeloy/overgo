//go:build windows

package executor

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/model"
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

type cudaReferenceFixture struct {
	t       *testing.T
	builder *tensor.Builder
	feeds   map[*tensor.Tensor]reference.Value
	seed    int
}

func newCUDAReferenceFixture(t *testing.T, seed int) *cudaReferenceFixture {
	t.Helper()
	cudatest.Require(t)
	return &cudaReferenceFixture{
		t: t, builder: tensor.NewBuilder(), feeds: make(map[*tensor.Tensor]reference.Value), seed: seed,
	}
}

func (f *cudaReferenceFixture) input(
	name string,
	shape tensor.Shape,
	scale, offset float32,
) *tensor.Tensor {
	item := f.builder.Input(name, dtype.F32, shape)
	f.feeds[item] = patternedValue(shape, f.seed, scale, offset)
	f.seed += 2
	return item
}

func (f *cudaReferenceFixture) nextInput(
	prefix string,
	shape tensor.Shape,
	scale, offset float32,
) *tensor.Tensor {
	return f.input(fmt.Sprintf("%s_%d", prefix, f.seed), shape, scale, offset)
}

func (f *cudaReferenceFixture) modelPlan(spec model.Spec) model.ModelPlan {
	f.t.Helper()
	plan, err := model.CompileModelPlan(spec, model.Weights{})
	if err != nil {
		f.t.Fatal(err)
	}
	return plan
}

func (f *cudaReferenceFixture) layer(
	program model.ModelPlan,
	layer int,
	spec model.Spec,
	weights model.LayerGraphWeights,
	context model.CachedBlockContext,
) model.DenseBlockResult {
	f.t.Helper()
	plan, err := program.Layer(layer)
	if err != nil {
		f.t.Fatal(err)
	}
	context.Builder, context.Layer = f.builder, plan.Layer
	result, err := model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan, Context: context,
	})
	if err != nil {
		f.t.Fatal(err)
	}
	return result
}

func (f *cudaReferenceFixture) requireMatch(outputs []*tensor.Tensor, tolerance float64) {
	f.t.Helper()
	checkCUDAGraph(f.t, f.feeds, uniformGraphChecks(outputs, tolerance)...)
}

func newFixtureExecutor(t testing.TB) *Executor {
	t.Helper()
	cudatest.Require(t)
	executor, err := New(cudaFixtureDevice)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "close CUDA fixture executor", executor.Close)
	return executor
}

func newFixtureWorker(t testing.TB) *device.Worker {
	t.Helper()
	cudatest.Require(t)
	worker, err := device.New(cudaFixtureDevice)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "close CUDA fixture worker", worker.Close)
	return worker
}

func newFixtureExecutorWithWorker(t testing.TB, worker *device.Worker) *Executor {
	t.Helper()
	executor, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "close CUDA fixture executor", executor.Close)
	return executor
}

func fixtureCleanup(t testing.TB, operation string, cleanup func() error) {
	t.Helper()
	t.Cleanup(func() {
		if err := cleanup(); err != nil {
			t.Errorf("%s: %v", operation, err)
		}
	})
}

func copyFixtureDeviceBytes(
	t testing.TB,
	worker *device.Worker,
	data []byte,
) driver.DevicePtr {
	t.Helper()
	var pointer driver.DevicePtr
	err := worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(data)))
		if allocateErr != nil {
			return allocateErr
		}
		if copyErr := state.Driver.MemcpyHtoD(pointer, data); copyErr != nil {
			freeErr := state.Driver.MemFree(pointer)
			pointer = 0
			return errors.Join(copyErr, freeErr)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "release CUDA fixture buffer", func() error {
		return worker.Do(context.Background(), func(state *device.State) error {
			return state.Driver.MemFree(pointer)
		})
	})
	return pointer
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

func buildFixtureCachedBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec model.Spec,
	weights model.LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layer uint32,
	recurrent bool,
) (model.DenseBlockResult, error) {
	plan := spec.PlanLayer(layer, recurrent)
	return model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, Positions: positions,
			PastKey: pastKey, PastValue: pastValue, Layer: layer, Recurrent: recurrent,
		},
		Spec: spec, Weights: weights, Plan: &plan,
	})
}

func buildFixtureDenseBlockCachedForLayer(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec model.Spec,
	weights model.LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layer uint32,
) (model.DenseBlockResult, error) {
	return buildFixtureCachedBlock(
		builder, input, spec, weights, positions, pastKey, pastValue, layer, false,
	)
}

func buildFixtureDenseBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec model.Spec,
	weights model.LayerGraphWeights,
	positions []uint32,
) (*tensor.Tensor, error) {
	result, err := buildFixtureDenseBlockCachedForLayer(
		builder, input, spec, weights, positions, nil, nil, 0,
	)
	return result.Output, err
}

func buildFixtureDenseBlockCachedWithMultiPositions(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec model.Spec,
	weights model.LayerGraphWeights,
	positions [4][]uint32,
	pastKey, pastValue *tensor.Tensor,
	layer uint32,
) (model.DenseBlockResult, error) {
	plan := spec.PlanLayer(layer, false)
	return model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, MultiPositions: &positions,
			PastKey: pastKey, PastValue: pastValue, Layer: layer,
		},
		Spec: spec, Weights: weights, Plan: &plan,
	})
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

func patternedValue(shape tensor.Shape, seed int, scale, bias float32) reference.Value {
	elements, _ := shape.Elements()
	data := make([]float32, int(elements))
	for index := range data {
		data[index] = bias + scale*float32(((index*7+seed*3)%19)-9)
	}
	value, _ := reference.NewValue(shape, data)
	return value
}

func compare(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for index := range want {
		if difference := math.Abs(float64(got[index] - want[index])); difference > tolerance {
			t.Fatalf("value[%d] = %v, want %v (difference %g)", index, got[index], want[index], difference)
		}
	}
}
