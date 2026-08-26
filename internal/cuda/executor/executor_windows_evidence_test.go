//go:build windows

package executor

import (
	"context"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestGraphReplayMetrics(t *testing.T) {
	ctx := context.Background()
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8)
	input := builder.Input("input", dtype.F32, shape)
	output := builder.Scale(input, 2)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	beforeDriver, err := cuda.worker.ExecutionStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: {Shape: shape, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
	}
	executions := graphExecCacheCapacity + 2
	for range executions {
		retained, executeErr := cuda.ExecuteRetainedCompiled(
			ctx, compiled, feeds, nil, nil, nil,
		)
		if executeErr != nil {
			t.Fatal(executeErr)
		}
		if releaseErr := retained.Release(ctx); releaseErr != nil {
			t.Fatal(releaseErr)
		}
	}
	metrics, err := cuda.Metrics(ctx)
	if err != nil {
		t.Fatal(err)
	}
	afterDriver, err := cuda.worker.ExecutionStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if metrics.GraphCacheHits == 0 || metrics.GraphCacheMisses == 0 {
		t.Fatalf("graph cache did not report both paths: %+v", metrics)
	}
	if metrics.GraphCaptures != metrics.GraphCacheMisses {
		t.Fatalf("captures = %d, misses = %d", metrics.GraphCaptures, metrics.GraphCacheMisses)
	}
	if metrics.GraphInstantiations+metrics.GraphUpdates != metrics.GraphCaptures {
		t.Fatalf("capture disposition is incomplete: %+v", metrics)
	}
	if metrics.GraphCacheEntries == 0 || metrics.GraphCacheEntries > metrics.GraphCacheCapacity {
		t.Fatalf("graph cache occupancy is invalid: %+v", metrics)
	}
	if launches := afterDriver.GraphLaunches - beforeDriver.GraphLaunches; launches != uint64(executions) {
		t.Fatalf("driver graph launches = %d, want %d", launches, executions)
	}
}

func TestArenaMetrics(t *testing.T) {
	ctx := context.Background()
	cuda := newFixtureExecutor(t)
	large := compileScaleFixture(t, tensor.MustShape(1024))
	small := compileScaleFixture(t, tensor.MustShape(4))
	if large.memory.ArenaSize <= small.memory.ArenaSize {
		t.Fatalf("fixture arena sizes = %d and %d", large.memory.ArenaSize, small.memory.ArenaSize)
	}
	if err := cuda.PrepareCompiled(ctx, large); err != nil {
		t.Fatal(err)
	}
	if err := cuda.PrepareCompiled(ctx, small); err != nil {
		t.Fatal(err)
	}
	metrics, err := cuda.Metrics(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if metrics.ArenaRequiredBytes != small.memory.ArenaSize ||
		metrics.ArenaPeakRequiredBytes != large.memory.ArenaSize ||
		metrics.ArenaCommittedBytes != large.memory.ArenaSize {
		t.Fatalf("arena byte evidence is invalid: %+v", metrics)
	}
	if metrics.ArenaUnusedBytes != metrics.ArenaCommittedBytes-metrics.ArenaRequiredBytes {
		t.Fatalf("arena unused bytes are invalid: %+v", metrics)
	}
	if metrics.ArenaGrowths == 0 {
		t.Fatalf("arena growth was not recorded: %+v", metrics)
	}
}

func TestExecutorReplayMatchesReference(t *testing.T) {
	ctx := context.Background()
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8)
	input := builder.Input("input", dtype.F32, shape)
	scaled := builder.Scale(input, 1.5)
	output := builder.Add(scaled, input)
	program, err := tensor.CompileProgram(output)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: {Shape: shape, Data: []float32{-4, -3, -2, -1, 1, 2, 3, 4}},
	}
	want, err := reference.Execute(program.Outputs(), feeds)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileProgram(program)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	for range graphExecCacheCapacity + 2 {
		retained, executeErr := cuda.ExecuteRetainedCompiled(
			ctx, compiled, feeds, nil, nil, nil,
		)
		if executeErr != nil {
			t.Fatal(executeErr)
		}
		got, copyErr := retained.CopyToHost(ctx, output)
		if copyErr != nil {
			t.Fatal(copyErr)
		}
		compare(t, got.Data, want[output].Data, accuracyExact)
		if releaseErr := retained.Release(ctx); releaseErr != nil {
			t.Fatal(releaseErr)
		}
	}
	metrics, err := cuda.Metrics(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if metrics.GraphCacheHits == 0 {
		t.Fatalf("parity exercised no replay: %+v", metrics)
	}
}

func compileScaleFixture(t testing.TB, shape tensor.Shape) *CompiledGraph {
	t.Helper()
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, shape)
	compiled, err := Compile(builder.Scale(input, 2))
	if err != nil {
		t.Fatal(err)
	}
	return compiled
}
