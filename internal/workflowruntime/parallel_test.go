package workflowruntime

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const (
	parallelBranchModule recipe.ModuleID = "branch"
	parallelJoinModule   recipe.ModuleID = "join"
	parallelLeftPort     recipe.PortName = "left"
	parallelRightPort    recipe.PortName = "right"
	parallelOutputPort   recipe.PortName = "output"
)

func TestJoinInputOrder(t *testing.T) {
	fixture := newParallelFixture(t)
	release := map[string]chan struct{}{"left": make(chan struct{}), "right": make(chan struct{})}
	started := make(chan string, len(release))
	registerParallelAdapters(t, fixture, func(ctx context.Context, value string) (string, error) {
		started <- value
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-release[value]:
			return value, nil
		}
	})
	done := make(chan Result, 1)
	errorsOut := make(chan error, 1)
	go func() {
		result, err := fixture.execute(context.Background())
		done <- result
		errorsOut <- err
	}()
	<-started
	<-started
	close(release["right"])
	close(release["left"])
	result := <-done
	if err := <-errorsOut; err != nil {
		t.Fatal(err)
	}
	if got := result.Outputs[parallelOutputPort].Items[0].Value; got != "left,right" {
		t.Fatalf("join output = %v", got)
	}
}

func TestResourceDerivedConcurrency(t *testing.T) {
	fixture := newParallelFixture(t)
	var active, maximum atomic.Int32
	release := make(chan struct{})
	started := make(chan struct{}, len(fixture.program.ReadySets()[0]))
	registerParallelAdapters(t, fixture, func(ctx context.Context, value string) (string, error) {
		current := active.Add(1)
		for observed := maximum.Load(); current > observed && !maximum.CompareAndSwap(observed, current); observed = maximum.Load() {
		}
		started <- struct{}{}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-release:
			active.Add(-1)
			return value, nil
		}
	})
	done := make(chan error, 1)
	go func() {
		_, err := fixture.execute(context.Background())
		done <- err
	}()
	for range fixture.program.ReadySets()[0] {
		<-started
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if maximum.Load() != int32(len(fixture.program.ReadySets()[0])) {
		t.Fatalf("maximum concurrency = %d", maximum.Load())
	}
}

func TestParallelCancellation(t *testing.T) {
	fixture := newParallelFixture(t)
	ready := make(chan struct{})
	var started atomic.Int32
	registerParallelAdapters(t, fixture, func(ctx context.Context, value string) (string, error) {
		if started.Add(1) == int32(len(fixture.program.ReadySets()[0])) {
			close(ready)
		}
		<-ready
		if value == "left" {
			return "", errors.New("branch failed")
		}
		<-ctx.Done()
		return "", ctx.Err()
	})
	if _, err := fixture.execute(context.Background()); err == nil || !strings.Contains(err.Error(), "branch failed") {
		t.Fatalf("parallel error = %v", err)
	}
}

func TestSingleLifecycleMutator(t *testing.T) {
	fixture := newParallelFixture(t)
	registerParallelAdapters(t, fixture, func(_ context.Context, value string) (string, error) { return value, nil })
	if _, err := fixture.execute(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, stage := range fixture.program.Stages() {
		receipt, found, err := runrecord.ResolveStageReceipt(context.Background(), fixture.store, fixture.operation, stage.Node.ID)
		if err != nil || !found || receipt.State != runrecord.StageCompleted || receipt.Attempt != 1 {
			t.Fatalf("stage %s receipt = (%+v, %t, %v)", stage.Node.ID, receipt, found, err)
		}
	}
}

func TestDeterministicParallelReceipts(t *testing.T) {
	for range 2 {
		fixture := newParallelFixture(t)
		registerParallelAdapters(t, fixture, func(_ context.Context, value string) (string, error) { return value, nil })
		result, err := fixture.execute(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if result.Outputs[parallelOutputPort].Items[0].Value != "left,right" {
			t.Fatalf("nondeterministic output = %+v", result.Outputs)
		}
	}
}

type parallelFixture struct {
	store     *repodb.Store
	program   recipe.Program
	runtime   *Runtime
	operation artifact.ID
}

func newParallelFixture(t *testing.T) parallelFixture {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	model := testutil.ArtifactID(t, artifact.KindModel, "parallel-model")
	if _, err := store.Commit(context.Background(), artifact.Batch{Key: "parallel/model", Artifacts: []artifact.Descriptor{{ID: model}}}); err != nil {
		t.Fatal(err)
	}
	branch := recipe.Module{
		ID: parallelBranchModule, Tasks: []recipe.Task{recipe.TaskInference}, Placements: []recipe.Placement{recipe.PlacementHost},
		Inputs:  []recipe.Port{{Name: "input", Data: recipe.DataText, Cardinality: recipe.CardinalityOne}},
		Outputs: []recipe.Port{{Name: parallelOutputPort, Data: recipe.DataText, Cardinality: recipe.CardinalityOne}},
	}
	join := recipe.Module{
		ID: parallelJoinModule, Tasks: []recipe.Task{recipe.TaskInference}, Placements: []recipe.Placement{recipe.PlacementHost},
		Inputs:  []recipe.Port{{Name: "inputs", Data: recipe.DataText, Cardinality: recipe.CardinalityMany}},
		Outputs: []recipe.Port{{Name: parallelOutputPort, Data: recipe.DataText, Cardinality: recipe.CardinalityOne}},
	}
	left := recipe.Node{ID: "left", Module: branch.ID, Placement: recipe.PlacementHost}
	right := recipe.Node{ID: "right", Module: branch.ID, Placement: recipe.PlacementHost}
	sink := recipe.Node{ID: "sink", Module: join.ID, Placement: recipe.PlacementHost}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskInference, []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: model}},
		[]recipe.Node{left, right, sink},
		[]recipe.Edge{
			{From: recipe.Endpoint{Node: left.ID, Port: parallelOutputPort}, To: recipe.Endpoint{Node: sink.ID, Port: "inputs"}},
			{From: recipe.Endpoint{Node: right.ID, Port: parallelOutputPort}, To: recipe.Endpoint{Node: sink.ID, Port: "inputs"}},
		},
		[]recipe.Input{
			{Name: parallelLeftPort, Data: recipe.DataText, Target: recipe.Endpoint{Node: left.ID, Port: "input"}},
			{Name: parallelRightPort, Data: recipe.DataText, Target: recipe.Endpoint{Node: right.ID, Port: "input"}},
		},
		[]recipe.Output{{Name: parallelOutputPort, Data: recipe.DataText, Source: recipe.Endpoint{Node: sink.ID, Port: parallelOutputPort}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := recipe.NewCatalog(branch, join)
	if err != nil {
		t.Fatal(err)
	}
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	operation, err := ExecutionID(definition.ID, "parallel")
	if err != nil {
		t.Fatal(err)
	}
	return parallelFixture{store: store, program: program, runtime: runtime, operation: operation}
}

func registerParallelAdapters(
	t *testing.T,
	fixture parallelFixture,
	branch func(context.Context, string) (string, error),
) {
	t.Helper()
	if err := RegisterContextStage(fixture.runtime, parallelBranchModule, fixture.program.Definition().Model,
		branch, textArtifact); err != nil {
		t.Fatal(err)
	}
	if err := fixture.runtime.Register(parallelJoinModule, AdapterFunc(
		func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			items := request.Inputs["inputs"].Items
			values := make([]string, len(items))
			for index, item := range items {
				values[index] = item.Value.(string)
			}
			joined := strings.Join(values, ",")
			content, err := textArtifact(joined)
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]Value{
				parallelOutputPort: ArtifactValue(recipe.DataText, joined, content),
			}, nil
		},
	)); err != nil {
		t.Fatal(err)
	}
}

func (fixture parallelFixture) execute(ctx context.Context) (Result, error) {
	return fixture.runtime.ExecuteProgram(ctx, "parallel", fixture.operation, fixture.program,
		map[recipe.PortName]Value{
			parallelLeftPort:  {Kind: recipe.DataText, Items: []Datum{{Value: "left"}}},
			parallelRightPort: {Kind: recipe.DataText, Items: []Datum{{Value: "right"}}},
		})
}

func textArtifact(value string) (artifact.Content, error) {
	return artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "overgo.parallel-text.v1"), value)
}
