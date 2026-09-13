package workflowruntime

import (
	"context"
	"errors"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/workflowrecipe"
)

func TestContextPipelineCarriesRequestContext(t *testing.T) {
	store, program := runtimeFixture(t)
	defer store.Close()
	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	type requestKey struct{}
	marker := new(int)
	ctx := context.WithValue(t.Context(), requestKey{}, marker)
	visited := 0
	check := func(ctx context.Context) error {
		if ctx.Value(requestKey{}) != marker {
			return errors.New("pipeline dropped request context")
		}
		visited++
		return nil
	}
	if err := RegisterContextPipeline(runtime, program.Definition().Model,
		workflowrecipe.ModuleTokenize, func(ctx context.Context, input string) ([]int, error) { return []int{len(input)}, check(ctx) },
		workflowrecipe.ModuleGenerate, func(ctx context.Context, input []int) (int, error) { return input[0], check(ctx) },
		workflowrecipe.ModuleDetokenize, func(ctx context.Context, input int) (string, error) { return strconv.Itoa(input), check(ctx) },
		func(value string) (artifact.Content, error) {
			return fixtureContent(t, artifact.KindOutput, value), nil
		},
	); err != nil {
		t.Fatal(err)
	}
	key := "runtime/context-pipeline"
	result, err := runtime.ExecuteProgram(ctx, key, runtimeExecutionID(t, program, key), nil, program, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Value: "request"}}},
	})
	if err != nil || visited != len(program.Stages()) {
		t.Fatalf("pipeline context: visited=%d error=%v", visited, err)
	}
	output, ok := result.Outputs["text"].Single()
	if !ok || output.Value != strconv.Itoa(len("request")) {
		t.Fatal("pipeline changed typed stage output")
	}
}

func TestScalarStageCancellationBoundaries(t *testing.T) {
	for _, boundary := range []string{"before-compute", "after-compute", "after-encode"} {
		t.Run(boundary, func(t *testing.T) {
			store, program := runtimeFixture(t)
			defer store.Close()
			runtime, err := NewForProgram(store, program)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			computed, encoded := false, false
			model := program.Definition().Model
			if err := RegisterResolvedStage(runtime, workflowrecipe.ModuleTokenize, model,
				func(context.Context, StepRequest) (string, error) {
					computed = true
					if boundary == "after-compute" {
						cancel(context.Canceled)
					}
					return "value", nil
				}, func(value string) (artifact.Content, error) {
					encoded = true
					if boundary == "after-encode" {
						cancel(context.Canceled)
					}
					return fixtureContent(t, artifact.KindOutput, value), nil
				}); err != nil {
				t.Fatal(err)
			}
			if boundary == "before-compute" {
				cancel(context.Canceled)
			}
			adapter, ok := runtime.adapter(workflowrecipe.ModuleTokenize)
			if !ok {
				t.Fatal("registered adapter missing")
			}
			outputs, err := adapter.Execute(ctx, StepRequest{Model: model})
			if !errors.Is(err, context.Canceled) || len(outputs) != 0 {
				t.Fatalf("canceled stage returned output: error=%v outputs=%d", err, len(outputs))
			}
			if computed != (boundary != "before-compute") || encoded != (boundary == "after-encode") {
				t.Fatalf("work crossed canceled boundary: computed=%t encoded=%t", computed, encoded)
			}
		})
	}
}
