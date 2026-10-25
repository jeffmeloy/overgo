//go:build windows

package executor

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
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

// lendingExperts is an expert owner with no banks that records the memory lent to it and the bytes taken back.
type lendingExperts struct {
	lent      map[driver.DevicePtr]uint64
	reclaimed map[driver.DevicePtr]uint64
}

func (*lendingExperts) IsExpertBank(string) bool { return false }
func (*lendingExperts) ExpertRoutes(*device.State, []*tensor.Tensor) (driver.DevicePtr, []byte, error) {
	return 0, nil, nil
}
func (*lendingExperts) ExpertsOnHost(*device.State, []*tensor.Tensor) (driver.DevicePtr, error) {
	return 0, nil
}
func (*lendingExperts) ExpertSources(*device.State, *tensor.Tensor, []driver.DevicePtr, [][]byte) error {
	return nil
}
func (*lendingExperts) AdvanceExperts(context.Context, *device.State, bool) error { return nil }
func (*lendingExperts) StartExpertCopies(context.Context, *device.State) error    { return nil }
func (*lendingExperts) ReserveExpertMemory(*device.State, uint64) error           { return nil }
func (*lendingExperts) ExpertMemory(*device.State) (uint64, error)                { return 0, nil }
func (l *lendingExperts) LendExpertMemory(_ context.Context, _ *device.State, pointer driver.DevicePtr, bytes uint64) error {
	l.lent[pointer] = bytes
	return nil
}
func (l *lendingExperts) ReclaimExpertMemory(_ context.Context, _ *device.State, pointer driver.DevicePtr, bytes uint64) error {
	l.reclaimed[pointer] = max(l.reclaimed[pointer], bytes)
	return nil
}

// TestPromptMemoryGivesBackItsLoan holds a read's loan to the experts on an executor that keeps its frames (a
// speculating runner's): a one-shot run, a prompt's read, takes an arena of its own, which no captured graph names;
// giving the loan back lends the arena and the matmul staging to the experts, freeing nothing and dropping no kept
// graph; the next read takes back the arena's bytes it uses before its run and a captured graph naming the staging
// takes all the staging back before its replay, its launches using every byte of it, and every run reads exactly as before. Without experts the memory stays
// the executor's.
func TestPromptMemoryGivesBackItsLoan(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.F32, tensor.MustShape(64, 1))
	right := builder.Input("right", dtype.F32, tensor.MustShape(64, 1))
	step := builder.Add(builder.Multiply(left, right), left)
	stepGraph, err := Compile(step)
	if err != nil {
		t.Fatal(err)
	}
	// The read's weight has twice the span's rows, so the staging it grows to is more than the span declares.
	staged := func(read bool) (*CompiledGraph, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
		builder := tensor.NewBuilder()
		rows := uint64(64)
		if read {
			rows *= 2
		}
		weight := builder.Input("weight", dtype.BF16, tensor.MustShape(64, rows))
		columns := builder.Input("columns", dtype.F32, tensor.MustShape(64, 8))
		output := builder.MulMat(weight, columns)
		if read {
			output = builder.Multiply(output, output)
		}
		graph, err := Compile(output)
		if err == nil {
			err = graph.ReserveExactWeightStaging()
		}
		if err != nil {
			t.Fatal(err)
		}
		return graph, weight, columns, output
	}
	spanGraph, spanWeight, spanColumns, span := staged(false)
	readGraph, readWeight, readColumns, read := staged(true)
	// One weight serves both: the span reads its first rows.
	storage, err := quant.Quantize(dtype.BF16, patternedValue(readWeight.Shape, 7, 0.25, 0).Data)
	if err != nil {
		t.Fatal(err)
	}
	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)
	if err := cuda.SetGraphExecCapacity(graphExecCacheCapacity + 4); err != nil {
		t.Fatal(err)
	}
	weights := func(graph *CompiledGraph, weight *tensor.Tensor) *DeviceInputs {
		inputs := graph.NewDeviceInputs()
		if err := inputs.Set(weight, pointer); err != nil {
			t.Fatal(err)
		}
		return inputs
	}
	spanInputs, readInputs := weights(spanGraph, spanWeight), weights(readGraph, readWeight)
	stepFeeds := map[*tensor.Tensor]reference.Value{left: patternedValue(left.Shape, 3, 0.5, 0), right: patternedValue(right.Shape, 5, 0.25, 0)}
	columns := patternedValue(spanColumns.Shape, 5, 0.5, 0)
	run := func(once bool, graph *CompiledGraph, feeds map[*tensor.Tensor]reference.Value, inputs *DeviceInputs, output *tensor.Tensor) []float32 {
		t.Helper()
		execute := cuda.ExecuteRetainedCompiled
		if once {
			execute = cuda.ExecuteRetainedCompiledOnce
		}
		retained, err := execute(t.Context(), graph, feeds, inputs, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		value, err := retained.CopyToHost(t.Context(), output)
		if err == nil {
			err = retained.Release(t.Context())
		}
		if err != nil {
			t.Fatal(err)
		}
		return value.Data
	}
	runSpan := func() []float32 {
		return run(false, spanGraph, map[*tensor.Tensor]reference.Value{spanColumns: columns}, spanInputs, span)
	}
	runRead := func() []float32 {
		return run(true, readGraph, map[*tensor.Tensor]reference.Value{readColumns: columns}, readInputs, read)
	}
	current := func() (bytes uint64) {
		t.Helper()
		if err := cuda.worker.Do(t.Context(), func(state *device.State) error {
			bytes = state.Driver.MemoryStats().CurrentBytes
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		return bytes
	}
	// The read runs first, growing the staging past the span's, and the span before the step: the span's feeds, the
	// larger, size the captured runs' feed buffer, and each growth drops the graphs that name what grew.
	readWant := runRead()
	spanWant := runSpan()
	stepWant := run(false, stepGraph, stepFeeds, nil, step)
	run(false, stepGraph, stepFeeds, nil, step)
	// Without experts to take it back, the read shares the captured runs' arena: a model without experts holds one.
	shared, err := cuda.Metrics(t.Context())
	if err != nil || shared.PromptArenaBytes != 0 || shared.BLASStagingBytes == 0 || shared.GraphCacheEntries != 2 {
		t.Fatalf("the read took an arena of %d and staging %d beside %d kept graphs (%v), want no arena of its own and the staging beside the step's and the span's", shared.PromptArenaBytes, shared.BLASStagingBytes, shared.GraphCacheEntries, err)
	}
	before := current()
	if err := cuda.ReleasePromptMemory(t.Context()); err != nil || current() != before {
		t.Fatalf("without experts giving the loan back changed %d bytes to %d (%v), want the memory kept", before, current(), err)
	}
	experts := &lendingExperts{lent: map[driver.DevicePtr]uint64{}, reclaimed: map[driver.DevicePtr]uint64{}}
	if err := cuda.SetExpertBanks(experts); err != nil {
		t.Fatal(err)
	}
	// Binding the experts drops the kept graphs, which the span and the step capture again.
	spanWant, stepWant = runSpan(), run(false, stepGraph, stepFeeds, nil, step)
	readWant = runRead()
	held, err := cuda.Metrics(t.Context())
	if err != nil || held.PromptArenaBytes == 0 {
		t.Fatalf("with experts the read took an arena of %d (%v), want its own", held.PromptArenaBytes, err)
	}
	before = current()
	arena, staging := cuda.resources.promptArena, cuda.resources.blas.staging
	if err := cuda.ReleasePromptMemory(t.Context()); err != nil {
		t.Fatal(err)
	}
	given, err := cuda.Metrics(t.Context())
	if err != nil || current() != before || given.GraphCacheEntries != 2 || experts.lent[arena] != held.PromptArenaBytes || experts.lent[staging] != held.BLASStagingBytes {
		t.Fatalf("lending left %d bytes of %d, %d kept graphs and lent %v (%v), want the arena and the staging lent and nothing freed or dropped", current(), before, given.GraphCacheEntries, experts.lent, err)
	}
	compare(t, run(false, stepGraph, stepFeeds, nil, step), stepWant, accuracyExact)
	if replayed, err := cuda.Metrics(t.Context()); err != nil || replayed.GraphCacheHits != given.GraphCacheHits+1 || replayed.GraphCaptures != given.GraphCaptures || len(experts.reclaimed) != 0 {
		t.Fatalf("the step ran with %d hits and %d captures after %d and %d, taking back %v (%v), want it replayed taking back nothing", replayed.GraphCacheHits, replayed.GraphCaptures, given.GraphCacheHits, given.GraphCaptures, experts.reclaimed, err)
	}
	// The span declares less staging than the read grew it to.
	compare(t, runSpan(), spanWant, accuracyExact)
	if experts.reclaimed[staging] != held.BLASStagingBytes {
		t.Fatalf("the span took back %d of the lent staging's %d bytes, want all of it, whose every byte its launches may use", experts.reclaimed[staging], held.BLASStagingBytes)
	}
	compare(t, runRead(), readWant, accuracyExact)
	if experts.reclaimed[arena] == 0 || experts.reclaimed[arena] > held.PromptArenaBytes {
		t.Fatalf("the next read took back %d of the arena's %d bytes, want the bytes it uses", experts.reclaimed[arena], held.PromptArenaBytes)
	}
}
