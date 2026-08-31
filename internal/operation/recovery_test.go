package operation

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
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
			id, err := manager.Submit(t.Context(), Request{Task: workflow.task, Recipe: recipeID},
				func(context.Context, Reporter) (Completion, error) {
					return Completion{Run: runID}, operatoraction.Recoverable(errors.New("artifact unavailable"), block)
				})
			if err != nil {
				t.Fatal(err)
			}
			status, err := manager.Wait(t.Context(), id)
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

	manager := newTestManager(t)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "subject-bound recovery")
	other := testutil.ArtifactID(t, artifact.KindRecipe, "different recovery subject")
	runID := testutil.ArtifactID(t, artifact.KindRun, "mismatched recovery run")
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskTraining, Recipe: recipeID},
		func(context.Context, Reporter) (Completion, error) {
			return Completion{Run: runID}, operatoraction.Recoverable(errors.New("wrong subject"), operatoraction.Block{
				Subject: other, Reason: "wrong recipe is unavailable", Evidence: []artifact.ID{runID},
				Actions: []operatoraction.Action{{Code: "retry", Summary: "Retry", Argv: []string{"overgo", "train"}}},
			})
		})
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	if status.State != StateFailed || status.Recovery != nil {
		t.Fatalf("cross-subject recovery was advertised: %+v", status)
	}
}

func TestOperationRecoveryReusesIdentity(t *testing.T) {
	manager := newTestManager(t)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "operation recovery recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "operation recovery run")
	request := Request{Task: recipe.TaskTraining, Recipe: recipeID}
	id, err := manager.Submit(t.Context(), request, func(context.Context, Reporter) (Completion, error) {
		return Completion{Run: runID}, errors.New("interrupted")
	})
	if err != nil {
		t.Fatal(err)
	}
	if status, err := manager.Wait(t.Context(), id); err != nil || status.State != StateFailed {
		t.Fatalf("initial status = (%+v, %v)", status, err)
	}
	if _, err := manager.Recover(t.Context(), id, Request{Task: recipe.TaskInference, Recipe: recipeID}, nil); err == nil {
		t.Fatal("mismatched recovery was admitted")
	}
	recovered, err := manager.Recover(t.Context(), id, request, func(context.Context, Reporter) (Completion, error) {
		return Completion{Run: runID}, nil
	})
	if err != nil || recovered != id {
		t.Fatalf("recovery = (%s, %v)", recovered, err)
	}
	status, err := manager.Wait(t.Context(), id)
	if err != nil || status.State != StateCompleted || status.Run == nil || *status.Run != runID {
		t.Fatalf("recovered status = (%+v, %v)", status, err)
	}
}

func TestBlockedOperationResumesAfterDecision(t *testing.T) {
	manager := newTestManager(t)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "decision recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "decision run")
	action := operatoraction.Action{Code: "resume", Summary: "Resume exact work", Argv: []string{"overgo", "resume"}}
	attempts := 0
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskTraining, Recipe: recipeID},
		func(context.Context, Reporter) (Completion, error) {
			attempts++
			if attempts == 1 {
				return Completion{Run: runID}, operatoraction.Recoverable(errors.New("approval required"), operatoraction.Block{
					Subject: recipeID, Reason: "operator approval required", Evidence: []artifact.ID{runID},
					Actions: []operatoraction.Action{action},
				})
			}
			return Completion{Run: runID}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if status, err := manager.Wait(t.Context(), id); err != nil || status.State != StateBlocked {
		t.Fatalf("blocked status = (%+v, %v)", status, err)
	}
	approval, err := operatoraction.NewApprovalRequest(id, recipeID, action, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := runrecord.NewHumanDecision(approval, operatoraction.AnswerGrant)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.RecoverAfterDecision(t.Context(), decision); err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(t.Context(), id)
	if err != nil || status.State != StateCompleted || attempts != 2 {
		t.Fatalf("recovered status = (%+v, %v), attempts=%d", status, err, attempts)
	}
	if _, err := manager.RecoverAfterDecision(t.Context(), decision); err == nil {
		t.Fatal("decision replay resumed completed work")
	}
}
