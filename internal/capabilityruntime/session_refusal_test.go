package capabilityruntime

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

// TestSessionThatCannotTakeARequestIsReplaced serves requests from a director
// whose resident session refuses some of them, as a session does that needs
// more beside what it holds than the device has left. A request the session
// takes is served by it and loads nothing. A request it refuses is served all
// the same: the session is closed and one is loaded for the request, as it
// would have been with none resident, and the answer is that request's. A
// request that was cancelled is not served by another session: it ends, and
// the session that refused it is closed.
func TestSessionThatCannotTakeARequestIsReplaced(t *testing.T) {
	t.Parallel()
	store, _, program := capabilityFixture(t, "refusing-session")
	loads, resets, closes := 0, 0, 0
	refusal := errors.New("the session cannot hold this request beside what it holds")
	var refuse func(context.Context, scalarRequest) error
	director, err := NewModelSessionDirector[scalarRequest, *cachedScalarModel, int](
		"scalar", "host", 1, func(scalarRequest) error { return nil },
		func(_ context.Context, _ artifact.Repository, _ string, _ recipe.Program, request scalarRequest) (*cachedScalarModel, error) {
			loads++
			return &cachedScalarModel{bias: request.Value, closed: &closes}, nil
		},
		func(ctx context.Context, model *cachedScalarModel, request scalarRequest) error {
			resets++
			if err := refuse(ctx, request); err != nil {
				return err
			}
			model.bias = request.Value
			return nil
		},
		func(runtime *workflowruntime.Runtime, bound artifact.ID, model *cachedScalarModel) error {
			return workflowruntime.RegisterJSONStage[scalarRequest, int](
				runtime, scalarModule, bound, artifact.JSONContract(artifact.KindOutput, "test.refusing-session-output.v1"),
				func(request scalarRequest) (int, error) { return request.Value + model.bias, nil },
			)
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer director.Close(t.Context())
	execute := director.Executor()
	execution := modelrecipetest.CandidateExecution(t, store, program)
	taken := func(context.Context, scalarRequest) error { return nil }
	refused := func(context.Context, scalarRequest) error { return refusal }
	for _, step := range []struct {
		what                  string
		raw                   string
		reset                 func(context.Context, scalarRequest) error
		want                  int
		loads, resets, closes int
	}{
		{"the first request", `{"value":4}`, taken, 8, 1, 0, 0},
		{"a request the session takes", `{"value":5}`, taken, 10, 1, 1, 0},
		{"a request the session refuses", `{"value":6}`, refused, 12, 2, 2, 1},
		{"a request the session loaded for that one takes", `{"value":7}`, taken, 14, 2, 3, 1},
		{"another it refuses", `{"value":8}`, refused, 16, 3, 4, 2},
	} {
		refuse = step.reset
		got, err := execute(t.Context(), store, "model", execution, step.raw)
		if err != nil || Unwrap(got) != step.want || loads != step.loads || resets != step.resets || closes != step.closes {
			t.Fatalf("%s: %v, %v, after %d loads, %d resets and %d closes; want %d after %d, %d and %d",
				step.what, Unwrap(got), err, loads, resets, closes, step.want, step.loads, step.resets, step.closes)
		}
	}
	// A cancelled request: the session's refusal is the cancellation's, and no other session is loaded for it.
	cancelled, cancel := context.WithCancel(t.Context())
	refuse = func(ctx context.Context, _ scalarRequest) error {
		cancel()
		return ctx.Err()
	}
	if _, err := execute(cancelled, store, "model", execution, `{"value":9}`); !errors.Is(err, context.Canceled) || loads != 3 || closes != 3 {
		t.Fatalf("a cancelled request ended with %v after %d loads and %d closes; want its cancellation, no load for it, and the session that held it closed", err, loads, closes)
	}
	// The request after it is served by a session of its own.
	refuse = taken
	if got, err := execute(t.Context(), store, "model", execution, `{"value":10}`); err != nil || Unwrap(got) != 20 || loads != 4 {
		t.Fatalf("after a cancelled request the next gave %v, %v after %d loads; want it served by a session loaded for it", Unwrap(got), err, loads)
	}
}
