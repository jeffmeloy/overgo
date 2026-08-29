package workflowcontract

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestBlockedWorkflowRecoveryJourneys(t *testing.T) {
	workflows := []struct {
		name   string
		task   recipe.Task
		code   string
		argv   []string
		result State
	}{
		{name: "training", task: recipe.TaskTraining, code: "resume-checkpoint", argv: []string{"overgo", "train", "--resume"}, result: StateRunning},
		{name: "evaluation", task: recipe.TaskInference, code: "materialize-dataset", argv: []string{"overgo", "evaluate", "--materialize"}, result: StateReady},
		{name: "serving", task: recipe.TaskGeneration, code: "retry-admission", argv: []string{"overgo", "serve", "--retry"}, result: StateReady},
	}
	for _, workflow := range workflows {
		t.Run(workflow.name, func(t *testing.T) {
			subject := testutil.ArtifactID(t, artifact.KindRecipe, workflow.name+" recipe")
			evidence := testutil.ArtifactID(t, artifact.KindEvidence, workflow.name+" blocker")
			block := operatoraction.Block{
				Subject: subject, Reason: workflow.name + " prerequisite is unavailable", Evidence: []artifact.ID{evidence},
				Actions: []operatoraction.Action{{Code: workflow.code, Summary: "Resolve the exact prerequisite", Argv: workflow.argv}},
			}
			start, err := FromOperation(workflow.name, operation.Status{
				Task: workflow.task, Recipe: subject, State: operation.StateBlocked, Recovery: &block,
			})
			if err != nil {
				t.Fatal(err)
			}
			registry := Registry{workflow.code: func(_ context.Context, current Snapshot, action operatoraction.Action) (Snapshot, error) {
				if action.Code != workflow.code || len(action.Argv) == 0 {
					t.Fatalf("handler received a different action: %+v", action)
				}
				return Snapshot{Workflow: current.Workflow, Subject: current.Subject, State: workflow.result}, nil
			}}
			transitions, err := ExerciseBlocked(t.Context(), start, registry)
			if err != nil || len(transitions) != 1 || transitions[0].To != workflow.result {
				t.Fatalf("journey transitions = (%+v, %v)", transitions, err)
			}

			if _, err := ExerciseBlocked(t.Context(), start, Registry{}); err == nil {
				t.Fatal("unregistered advertised action passed journey verification")
			}
			deadEnd := Registry{workflow.code: func(_ context.Context, current Snapshot, _ operatoraction.Action) (Snapshot, error) {
				return current, nil
			}}
			if _, err := ExerciseBlocked(t.Context(), start, deadEnd); err == nil {
				t.Fatal("self-looping recovery passed journey verification")
			}
			wrongSubject := operation.Status{Task: workflow.task, Recipe: subject, State: operation.StateBlocked, Recovery: &block}
			wrongSubject.Recovery = new(block.Clone())
			wrongSubject.Recovery.Subject = testutil.ArtifactID(t, artifact.KindRecipe, workflow.name+" other recipe")
			if _, err := FromOperation(workflow.name, wrongSubject); err == nil {
				t.Fatal("recovery for a different subject entered the journey")
			}
		})
	}
}
