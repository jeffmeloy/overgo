package server

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

type lifecycleTestReporter struct{ id artifact.ID }

func (reporter lifecycleTestReporter) OperationID() artifact.ID { return reporter.id }
func (reporter lifecycleTestReporter) Attempt(artifact.ID)      {}
func (lifecycleTestReporter) Progress(uint64, *uint64)          {}
func (lifecycleTestReporter) Metric(operation.Metric)           {}
func (lifecycleTestReporter) Publishing()                       {}

func TestTrainingAndEvaluationLifecycleReentry(t *testing.T) {
	requests := []operation.Request{
		{Task: recipe.TaskTraining, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "lifecycle training recipe")},
		{Task: recipe.TaskInference, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "lifecycle evaluation recipe")},
	}
	for _, request := range requests {
		t.Run(string(request.Task), func(t *testing.T) {
			ctx := context.Background()
			root := filepath.Join(t.TempDir(), "store")
			taskName := string(request.Task)
			operationID := testutil.ArtifactID(t, artifact.KindEvidence, taskName+" operation")
			intent := testutil.ArtifactID(t, artifact.KindEvidence, taskName+" intent")
			run := testutil.ArtifactID(t, artifact.KindRun, taskName+" run")
			output := testutil.ArtifactID(t, artifact.KindOutput, taskName+" output")
			var executed bool
			execute := func(repository artifact.Repository) func(context.Context) (operation.Completion, error) {
				return func(ctx context.Context) (operation.Completion, error) {
					if executed {
						return operation.Completion{}, errors.New("lifecycle execution repeated")
					}
					executed = true
					_, err := repository.Commit(ctx, artifact.Batch{
						Key:       "fixture/lifecycle/" + run.String(),
						Artifacts: []artifact.Descriptor{{ID: run}, {ID: output}},
					})
					return operation.Completion{Run: run, Outputs: []artifact.ID{output}}, err
				}
			}

			store, err := overgodb.Open(root)
			if err != nil {
				t.Fatal(err)
			}
			first, err := operation.ExecuteReentrant(ctx, store, lifecycleTestReporter{operationID}, request, intent, execute(store))
			if err != nil {
				t.Fatal(err)
			}
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}

			store, err = overgodb.Open(root)
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			second, err := operation.ExecuteReentrant(ctx, store, lifecycleTestReporter{operationID}, request, intent, execute(store))
			if err != nil || second.Run != first.Run || !slices.Equal(second.Outputs, first.Outputs) || !executed {
				t.Fatalf("reentry = (%+v, %v, executed=%v), first=%+v", second, err, executed, first)
			}
			otherIntent := testutil.ArtifactID(t, artifact.KindEvidence, taskName+" other intent")
			if _, err := operation.ExecuteReentrant(ctx, store, lifecycleTestReporter{operationID}, request, otherIntent, execute(store)); !errors.Is(err, operation.ErrLifecycleConflict) {
				t.Fatalf("changed intent error = %v", err)
			}
		})
	}
}
