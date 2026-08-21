package operation

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestRecoverableOperationStatus(t *testing.T) {
	workflows := []struct {
		name string
		task recipe.Task
	}{{name: "training", task: recipe.TaskTraining}, {name: "evaluation", task: recipe.TaskInference}}
	for _, workflow := range workflows {
		t.Run(workflow.name, func(t *testing.T) {
			manager := newTestManager(t)
			recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recoverable "+workflow.name)
			runID := testutil.ArtifactID(t, artifact.KindRun, "blocked "+workflow.name)
			block := operatoraction.Block{
				Subject: recipeID, Reason: "required artifact is unavailable", Evidence: []artifact.ID{runID},
				Actions: []operatoraction.Action{{Code: "retry", Summary: "Retry the exact recipe", Argv: []string{"overgo", workflow.name, "--recipe", recipeID.String()}}},
			}
			id, err := manager.Submit(context.Background(), Request{Task: workflow.task, Recipe: recipeID},
				func(context.Context, Reporter) (Completion, error) {
					return Completion{Run: runID}, operatoraction.Recoverable(errors.New("artifact unavailable"), block)
				})
			if err != nil {
				t.Fatal(err)
			}
			status, err := manager.Wait(context.Background(), id)
			if err != nil {
				t.Fatal(err)
			}
			if status.State != StateBlocked || status.Run == nil || *status.Run != runID || status.Recovery == nil ||
				status.Recovery.Subject != recipeID || status.Recovery.Actions[0].Argv[2] != "--recipe" {
				t.Fatalf("recoverable status = %+v", status)
			}
			status.Recovery.Actions[0].Argv[0] = "mutated"
			again, _ := manager.Status(id)
			if again.Recovery.Actions[0].Argv[0] != "overgo" {
				t.Fatal("status clone exposed mutable recovery argv")
			}
		})
	}
}
