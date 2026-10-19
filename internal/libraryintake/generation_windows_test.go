//go:build windows

package libraryintake

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type imageValidationReporter struct {
	completed, total uint64
	publishing       bool
}

func (*imageValidationReporter) OperationID() artifact.ID { return artifact.ID{} }
func (r *imageValidationReporter) Progress(v uint64, total *uint64) {
	r.completed = v
	if total != nil {
		r.total = *total
	}
}
func (*imageValidationReporter) Metric(operation.Metric) {}
func (*imageValidationReporter) Attempt(artifact.ID)     {}
func (*imageValidationReporter) Text(string)             {}
func (r *imageValidationReporter) Publishing()           { r.publishing = true }

func TestMOSSImageValidation(t *testing.T) {
	cudatest.Require(t)
	for _, mode := range []string{"success", "different-repeat", "cancel", "unmeasured"} {
		t.Run(mode, func(t *testing.T) {
			root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			capability := mediacapability.Catalog[recipe.TaskImageGen]
			capability.Execute = capabilityruntime.JSONScalar[latentimage.Request, *latentimage.Generator, latentimage.EncodedImage](
				"image-gen", latentimage.ValidateRequest,
				func(ctx context.Context, repository artifact.Repository, path string, program recipe.Program, input latentimage.Request) (*latentimage.Generator, error) {
					id, ok := program.Definition().PrimaryDependency(recipe.DependencyProfile)
					if !ok {
						return nil, errors.New("fixture profile missing")
					}
					profile, err := latentimage.ReadProfile(ctx, repository, id)
					if err != nil {
						return nil, err
					}
					return latentimage.LoadGenerator(ctx, path, profile, input)
				}, latentimage.RegisterRuntime)
			original := capability.Execute
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			calls := 0
			capability.Execute = func(ctx context.Context, repo artifact.Repository, path string, execution modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
				calls++
				if mode == "unmeasured" {
					return "invalid output", nil
				}
				if calls == 2 && mode == "cancel" {
					cancel()
				}
				if calls == 2 && mode == "different-repeat" {
					var value map[string]any
					if err := json.Unmarshal([]byte(input), &value); err != nil {
						return nil, err
					}
					value["seed"] = 1
					b, err := json.Marshal(value)
					if err != nil {
						return nil, err
					}
					input = string(b)
				}
				return original(ctx, repo, path, execution, input)
			}
			request, execute, err := prepareGenerationValidation(ctx, store, root, recipe.TaskImageGen, capability, "0123456789abcdef0123456789abcdef01234567", "blue cube")
			if err != nil {
				t.Fatal(err)
			}
			id := request.Recipe
			if request.Task != recipe.TaskImageGen {
				t.Fatal(request)
			}
			definition, err := recipe.RequireDefinition(t.Context(), store, id)
			if err != nil {
				t.Fatal(err)
			}
			status, present, err := modelrecipe.Status(ctx, store, id)
			if err != nil || !present || status != recipe.StatusCandidate {
				t.Fatal("premature activation", status, err)
			}
			reporter := &imageValidationReporter{}
			completion, executeErr := execute(ctx, reporter)
			status, present, err = modelrecipe.Status(t.Context(), store, id)
			if err != nil || !present {
				t.Fatal(status, err)
			}
			if mode == "success" {
				if executeErr != nil || calls != 2 || status != recipe.StatusActive || len(completion.Outputs) != 2 || !reporter.publishing || reporter.completed != reporter.total {
					t.Fatal("validation did not activate exact repeat", executeErr, calls, status, completion, reporter)
				}
				run, err := runrecord.RequireExactRun(t.Context(), store, completion.Run)
				if err != nil || run.Outcome != runrecord.OutcomeSucceeded || len(run.Inputs) != 1 {
					t.Fatal("verification input or success missing", run, err)
				}
				selection, err := modelrecipe.ResolveActiveExecution(t.Context(), store, definition.Model, recipe.TaskImageGen, modelrecipe.SessionWarm)
				if err != nil || selection.Program.Definition().ID != id {
					t.Fatal("activation cannot serve", err)
				}
			} else {
				if executeErr == nil || status == recipe.StatusActive || reporter.publishing || len(completion.Outputs) != 0 {
					t.Fatal("failed validation became active", executeErr, status, completion)
				}
				if mode == "cancel" && !errors.Is(executeErr, context.Canceled) {
					t.Fatal("cancellation lost", executeErr)
				}
				run, err := runrecord.RequireExactRun(t.Context(), store, completion.Run)
				if err != nil || run.Outcome != runrecord.OutcomeFailed {
					t.Fatal("failed validation not retained", executeErr, err)
				}
			}
		})
	}
}
