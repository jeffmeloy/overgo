//go:build windows

package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

func TestArenaMetrics(t *testing.T) {
	ctx := t.Context()
	cuda := newFixtureExecutor(t)
	large := compileScaleFixture(t, tensor.MustShape(1024))
	small := compileScaleFixture(t, tensor.MustShape(4))
	// The run's graph holds its scaled input in the arena, its sum retained.
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1<<16))
	huge, err := Compile(builder.Add(builder.Scale(input, 2), input))
	if err != nil {
		t.Fatal(err)
	}
	if large.memory.ArenaSize <= small.memory.ArenaSize {
		t.Fatalf("fixture arena sizes = %d and %d", large.memory.ArenaSize, small.memory.ArenaSize)
	}
	// A first arena takes its need exactly, and so does a prepared graph's growth, its need planned; a run that grows it
	// after rounds its need up to the granule.
	for growths, prepared := range []*CompiledGraph{small, large} {
		if err := cuda.PrepareCompiled(ctx, prepared); err != nil {
			t.Fatal(err)
		}
		metrics, err := cuda.Metrics(ctx)
		if err != nil || metrics.ArenaCommittedBytes != prepared.memory.ArenaSize || metrics.ArenaGrowths != uint64(growths+1) {
			t.Fatalf("a prepared arena committed %d bytes over %d growths (%v), want the graph's need %d exactly", metrics.ArenaCommittedBytes, metrics.ArenaGrowths, err, prepared.memory.ArenaSize)
		}
	}
	retained, err := cuda.ExecuteRetainedCompiledOnce(ctx, huge, map[*tensor.Tensor]reference.Value{
		input: {Shape: input.Shape, Data: make([]float32, input.Shape.Dims[0])},
	}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := retained.Release(ctx); err != nil {
		t.Fatal(err)
	}
	metrics, err := cuda.Metrics(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if metrics.ArenaRequiredBytes <= large.memory.ArenaSize || metrics.ArenaPeakRequiredBytes != metrics.ArenaRequiredBytes ||
		metrics.ArenaCommittedBytes != arenaBytes(metrics.ArenaRequiredBytes) || metrics.ArenaGrowths != 3 {
		t.Fatalf("arena byte evidence is invalid: %+v", metrics)
	}
	if metrics.ArenaUnusedBytes != metrics.ArenaCommittedBytes-metrics.ArenaRequiredBytes {
		t.Fatalf("arena unused bytes are invalid: %+v", metrics)
	}
	// A smaller graph prepared after the growth needs its own bytes and keeps the grown arena, its peak and its growths.
	grown := metrics
	testcheck.NoError(t, cuda.PrepareCompiled(ctx, small))
	metrics = testcheck.Must(cuda.Metrics(ctx))(t)
	if metrics.ArenaRequiredBytes != small.memory.ArenaSize || metrics.ArenaPeakRequiredBytes != grown.ArenaPeakRequiredBytes ||
		metrics.ArenaCommittedBytes != grown.ArenaCommittedBytes || metrics.ArenaGrowths != grown.ArenaGrowths ||
		metrics.ArenaUnusedBytes != metrics.ArenaCommittedBytes-metrics.ArenaRequiredBytes {
		t.Fatalf("a smaller graph after the growth left arena evidence %+v, after %+v", metrics, grown)
	}
}

func TestExecutorReplayMatchesReference(t *testing.T) {
	ctx := t.Context()
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
	beforeDriver := testcheck.Must(cuda.worker.ExecutionStats(ctx))(t)
	executions := graphExecCacheCapacity + 2
	for range executions {
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
	metrics := testcheck.Must(cuda.Metrics(ctx))(t)
	afterDriver := testcheck.Must(cuda.worker.ExecutionStats(ctx))(t)
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
