package capabilityruntime

import (
	"context"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/workflowruntime"
)

// TestReplayDoesNotCrossACodeChange holds a stored answer to the code that
// gave it, on both executor shapes. A request's answer is kept under the
// recipe and the request, and a recipe's identity does not move when the code
// that computes it is mended: a request first served by faulty code was
// answered with the faulty answer ever after. Here two builds serve one
// request from one store, the second computing another answer. Each build's
// first serving runs the model's node and gives that build's answer; each
// build's repeat is answered from the store with no model loaded and no node
// run; and the first build, serving again after the second, still replays its
// own.
func TestReplayDoesNotCrossACodeChange(t *testing.T) {
	// Not parallel: the package's code scope is this test's to move, and the tests that run in parallel wait for it.
	built := codeScope
	defer func() { codeScope = built }()
	validate := func(scalarRequest) error { return nil }
	for _, name := range []string{"scalar", "session"} {
		t.Run(name, func(t *testing.T) {
			var loads, runs atomic.Int32
			bias := 0
			load := func(context.Context, artifact.Repository, string, recipe.Program, scalarRequest) (int, error) {
				loads.Add(1)
				return 0, nil
			}
			bind := func(runtime *workflowruntime.Runtime, bound artifact.ID, _ int) error {
				return workflowruntime.RegisterJSONStage[scalarRequest, int](
					runtime, scalarModule, bound, artifact.JSONContract(artifact.KindOutput, "test.code-output.v1"),
					func(request scalarRequest) (int, error) {
						runs.Add(1)
						return request.Value + bias, nil
					},
				)
			}
			execute := JSONScalar[scalarRequest, int, int]("scalar", validate, load, bind)
			if name == "session" {
				director := testcheck.Must(NewModelSessionDirector[scalarRequest, int, int](
					"scalar", "host", 1, validate, load, func(context.Context, int, scalarRequest) error { return nil }, bind,
				))(t)
				defer director.Close(t.Context())
				execute = director.Executor()
			}
			store, _, program := capabilityFixture(t, "code-"+name)
			execution := modelrecipetest.CandidateExecution(t, store, program)
			for index, step := range []struct {
				build string
				bias  int
				runs  int32
			}{{"commit/first", 1, 1}, {"commit/first", 1, 1}, {"commit/mended", 2, 2}, {"commit/mended", 2, 2}, {"commit/first", 1, 2}} {
				codeScope, bias = func() string { return step.build }, step.bias
				loaded, ran := loads.Load(), runs.Load()
				got := testcheck.Must(execute(t.Context(), store, "model", execution, `{"value":4}`))(t)
				measured := got.(Measured)
				replayed := runs.Load() == ran
				if measured.Output != 4+step.bias || runs.Load() != step.runs || replayed && (loads.Load() != loaded || len(measured.Phases) != 0 || measured.Load != 0) {
					t.Fatalf("serving %d by build %s gave %v after %d runs of the node and %d loads (%d before), phases %v; want %d after %d runs, and a replay to load and run nothing",
						index, step.build, measured.Output, runs.Load(), loads.Load(), loaded, measured.Phases, 4+step.bias, step.runs)
				}
			}
		})
	}
}

// TestABoundFileKeysItsRun holds a director's served runs to the content it binds to each request (BindContent): a
// request whose bound content changed is computed afresh, and the earlier content again is answered from the store
// with no node run. A picture request's adapter replaced under its name had replayed the earlier file's picture
// (SQA-406).
func TestABoundFileKeysItsRun(t *testing.T) {
	t.Parallel()
	var runs atomic.Int32
	bound := "first"
	validate := func(scalarRequest) error { return nil }
	load := func(context.Context, artifact.Repository, string, recipe.Program, scalarRequest) (int, error) {
		return 0, nil
	}
	bind := func(runtime *workflowruntime.Runtime, model artifact.ID, _ int) error {
		return workflowruntime.RegisterJSONStage[scalarRequest, int](
			runtime, scalarModule, model, artifact.JSONContract(artifact.KindOutput, "test.bound-output.v1"),
			func(request scalarRequest) (int, error) { runs.Add(1); return request.Value, nil },
		)
	}
	director := testcheck.Must(NewModelSessionDirector[scalarRequest, int, int](
		"scalar", "host", 1, validate, load, func(context.Context, int, scalarRequest) error { return nil }, bind,
	))(t).BindContent(func(context.Context, string, scalarRequest) (string, error) { return bound, nil })
	defer director.Close(t.Context())
	store, _, program := capabilityFixture(t, "bound")
	execution := modelrecipetest.CandidateExecution(t, store, program)
	for index, step := range []struct {
		bound string
		runs  int32
	}{{"first", 1}, {"first", 1}, {"replaced", 2}, {"first", 2}} {
		bound = step.bound
		testcheck.Must(director.Executor()(t.Context(), store, "model", execution, `{"value":4}`))(t)
		if runs.Load() != step.runs {
			t.Fatalf("serving %d with %s bound ran the node %d times; want %d, a changed binding run afresh and a binding seen before replayed", index, step.bound, runs.Load(), step.runs)
		}
	}
}
