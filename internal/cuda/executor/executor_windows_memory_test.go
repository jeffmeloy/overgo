//go:build windows

package executor

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestExecutorRetainedOutputLifetime(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.Add(left, right)
	leftValue, _ := reference.NewValue(shape, []float32{1, 2, 3, 4})
	rightValue, _ := reference.NewValue(shape, []float32{10, 20, 30, 40})
	cuda := newFixtureExecutor(t)
	feeds := map[*tensor.Tensor]reference.Value{left: leftValue, right: rightValue}
	if _, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds); err != nil {
		t.Fatal(err)
	}
	before, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	retained, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		feeds,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	during, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if during.CurrentBytes-before.CurrentBytes != minimumDeviceBufferBytes {
		t.Fatalf(
			"retained pool bytes = %d, want %d",
			during.CurrentBytes-before.CurrentBytes,
			minimumDeviceBufferBytes,
		)
	}
	got, err := retained.CopyToHost(context.Background(), output)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got.Data, []float32{11, 22, 33, 44}, accuracyExact)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := retained.Release(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled release error = %v", err)
	}
	if _, ok := retained.Value(output); !ok {
		t.Fatal("canceled release discarded retained output")
	}
	if err := retained.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := retained.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	after, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if after.CurrentBytes != during.CurrentBytes {
		t.Fatalf("pooled bytes after release = %d, want %d", after.CurrentBytes, during.CurrentBytes)
	}
	if _, ok := retained.Value(output); ok {
		t.Fatal("released output remains accessible")
	}
	reused, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		feeds,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	reusedStats, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if reusedStats.Allocations != after.Allocations || reusedStats.CurrentBytes != after.CurrentBytes {
		t.Fatalf("retained pool did not reuse allocation: before=%+v after=%+v", after, reusedStats)
	}
	if err := reused.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestExecutorRetainedFlatSlicesShareProducerStorage(t *testing.T) {
	cudatest.Require(t)
	const (
		firstOffset  = 1
		firstLength  = 3
		secondOffset = 5
		secondLength = 2
	)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8)
	input := builder.Input("input", dtype.F32, shape)
	producer := builder.Scale(input, 2)
	first := builder.FlatSlice(producer, firstOffset, firstLength)
	second := builder.FlatSlice(producer, secondOffset, secondLength)
	cuda := newFixtureExecutor(t)
	retained, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{first, second},
		map[*tensor.Tensor]reference.Value{
			input: {Shape: shape, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "release retained fixture slices", func() error {
		return retained.Release(context.Background())
	})
	firstValue, firstOK := retained.Value(first)
	secondValue, secondOK := retained.Value(second)
	wantPointerDelta := fixtureShapeBytes(
		t, tensor.MustShape(secondOffset-firstOffset), dtype.F32,
	)
	if !firstOK || !secondOK || uint64(secondValue.Pointer-firstValue.Pointer) != wantPointerDelta {
		t.Fatalf("retained slice pointers = %+v/%+v", firstValue, secondValue)
	}
	firstHost, err := retained.CopyToHost(context.Background(), first)
	if err != nil {
		t.Fatal(err)
	}
	secondHost, err := retained.CopyToHost(context.Background(), second)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, firstHost.Data, []float32{4, 6, 8}, accuracyExact)
	compare(t, secondHost.Data, []float32{12, 14}, accuracyExact)
}

func TestExecutorStableTargetAppendsWithoutPrefixCopy(t *testing.T) {
	cudatest.Require(t)
	cuda := newFixtureExecutor(t)
	capacityShape := tensor.MustShape(8)
	buffer, err := cuda.AllocateDeviceBuffer(
		context.Background(), fixtureShapeBytes(t, capacityShape, dtype.F32),
	)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "release append fixture buffer", func() error {
		return buffer.Release(context.Background())
	})

	initialBuilder := tensor.NewBuilder()
	initialShape := tensor.MustShape(3)
	initialInput := initialBuilder.Input("initial", dtype.F32, initialShape)
	initialOutput := initialBuilder.Scale(initialInput, 1)
	initialTarget, err := buffer.Value(initialShape)
	if err != nil {
		t.Fatal(err)
	}
	initialGraph, err := Compile(initialOutput)
	if err != nil {
		t.Fatal(err)
	}
	initialTargets := initialGraph.NewRetainedTargets()
	if err := initialTargets.Set(initialOutput, initialTarget); err != nil {
		t.Fatal(err)
	}
	initial, err := cuda.ExecuteRetainedCompiled(
		context.Background(),
		initialGraph,
		map[*tensor.Tensor]reference.Value{
			initialInput: {Shape: initialShape, Data: []float32{1, 2, 3}},
		},
		nil, initialTargets, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := initial.Release(context.Background()); err != nil {
		t.Fatal(err)
	}

	appendBuilder := tensor.NewBuilder()
	past := appendBuilder.Input("past", dtype.F32, initialShape)
	newShape := tensor.MustShape(2)
	added := appendBuilder.Input("added", dtype.F32, newShape)
	joined := appendBuilder.Concat(past, added, 0)
	joinedTarget, err := buffer.Value(joined.Shape)
	if err != nil {
		t.Fatal(err)
	}
	appendGraph, err := Compile(joined)
	if err != nil {
		t.Fatal(err)
	}
	appendTargets := appendGraph.NewRetainedTargets()
	if err := appendTargets.Set(joined, joinedTarget); err != nil {
		t.Fatal(err)
	}
	appendInputs, err := appendGraph.BindDeviceInputs(map[*tensor.Tensor]driver.DevicePtr{past: initialTarget.Pointer})
	if err != nil {
		t.Fatal(err)
	}
	retained, err := cuda.ExecuteRetainedCompiled(
		context.Background(),
		appendGraph,
		map[*tensor.Tensor]reference.Value{
			added: {Shape: newShape, Data: []float32{4, 5}},
		},
		appendInputs, appendTargets, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "release appended fixture output", func() error {
		return retained.Release(context.Background())
	})
	got, err := retained.CopyToHost(context.Background(), joined)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got.Data, []float32{1, 2, 3, 4, 5}, accuracyExact)
}

func TestExecutorRejectsUndeclaredRetainedTargetAlias(t *testing.T) {
	cudatest.Require(t)
	cuda := newFixtureExecutor(t)
	shape := tensor.MustShape(4)
	buffer, err := cuda.AllocateDeviceBuffer(
		context.Background(), fixtureShapeBytes(t, shape, dtype.F32),
	)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "release alias fixture buffer", func() error {
		return buffer.Release(context.Background())
	})
	value, err := buffer.Value(shape)
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, shape)
	output := builder.Scale(input, 2)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	targets := compiled.NewRetainedTargets()
	if err := targets.Set(output, value); err != nil {
		t.Fatal(err)
	}
	inputs, err := compiled.BindDeviceInputs(map[*tensor.Tensor]driver.DevicePtr{input: value.Pointer})
	if err != nil {
		t.Fatal(err)
	}
	_, err = cuda.ExecuteRetainedCompiled(context.Background(), compiled, nil, inputs, targets, nil)
	if err == nil || !strings.Contains(err.Error(), "overlaps input") {
		t.Fatalf("undeclared target alias error = %v", err)
	}
}

func TestReleaseDeviceBuffersIsAtomicAndRetryable(t *testing.T) {
	cudatest.Require(t)
	cuda := newFixtureExecutor(t)
	shape := tensor.MustShape(4)
	bufferBytes := fixtureShapeBytes(t, shape, dtype.F32)
	first, err := cuda.AllocateDeviceBuffer(context.Background(), bufferBytes)
	if err != nil {
		t.Fatal(err)
	}
	second, err := cuda.AllocateDeviceBuffer(context.Background(), bufferBytes)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := ReleaseDeviceBuffers(ctx, first, second); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled release error = %v", err)
	}
	if _, err := first.Value(shape); err != nil {
		t.Fatalf("first buffer mutated by canceled release: %v", err)
	}
	if _, err := second.Value(shape); err != nil {
		t.Fatalf("second buffer mutated by canceled release: %v", err)
	}
	if err := ReleaseDeviceBuffers(context.Background(), first, second); err != nil {
		t.Fatal(err)
	}
	if _, err := first.Value(shape); err == nil {
		t.Fatal("released first buffer remains accessible")
	}
	if _, err := second.Value(shape); err == nil {
		t.Fatal("released second buffer remains accessible")
	}
}

func TestExecutorCopyDeviceValuesConcatenatesSegments(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	input := builder.Input("input", dtype.F32, shape)
	output := builder.Scale(input, 1)
	inputValue, _ := reference.NewValue(shape, []float32{1, 2, 3, 4})
	cuda := newFixtureExecutor(t)
	retained, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "release copied fixture source", func() error {
		return retained.Release(context.Background())
	})
	source, ok := retained.Value(output)
	if !ok {
		t.Fatal("retained output is unavailable")
	}
	segmentShape := tensor.MustShape(2)
	segmentBytes := fixtureShapeBytes(t, segmentShape, dtype.F32)
	copiedOwner, copied, err := cuda.CopyDeviceValues(
		context.Background(),
		[]DeviceCopy{{
			Shape: shape,
			Segments: []DeviceCopySegment{
				{Source: source.Pointer + driver.DevicePtr(segmentBytes), Bytes: segmentBytes},
				{Source: source.Pointer, Bytes: segmentBytes},
			},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	fixtureCleanup(t, "release copied fixture owner", func() error {
		return copiedOwner.Release(context.Background())
	})
	data := make([]float32, 4)
	err = cuda.worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemcpyDtoH(driver.Bytes(data), copied[0].Pointer)
	})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, data, []float32{3, 4, 1, 2}, accuracyExact)
}
