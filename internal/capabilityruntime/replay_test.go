package capabilityruntime

import (
	"context"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
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
				director, err := NewModelSessionDirector[scalarRequest, int, int](
					"scalar", "host", 1, validate, load, func(context.Context, int, scalarRequest) error { return nil }, bind,
				)
				if err != nil {
					t.Fatal(err)
				}
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
				got, err := execute(t.Context(), store, "model", execution, `{"value":4}`)
				if err != nil {
					t.Fatal(err)
				}
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
