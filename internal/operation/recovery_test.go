package operation

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"

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
					return Completion{Run: runID}, testutil.Recoverable(errors.New("artifact unavailable"), block)
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
			return Completion{Run: runID}, testutil.Recoverable(errors.New("wrong subject"), operatoraction.Block{
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

// TestTerminalTransitionIsAtomic holds an attempt's terminal transition to one
// locked step: when the observer sees the terminal status, the attempt's
// cancellation is spent and its waiters are released, and that status is
// published once. A recovery made the moment the terminal status is observed
// runs under its own cancellation; a waiter attached to the first attempt
// returns the first attempt's outcome, and one attached to the recovery
// returns the recovery's. Under a retention limit of one, a waiter attached
// before a later submission evicts its entry still returns that entry's
// outcome.
func TestTerminalTransitionIsAtomic(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		manager := newTestManager(t)
		terminals := make(chan Status, 4)
		manager.Notify(func(event Event) {
			if !terminal(event.Status.State) {
				return
			}
			current := manager.entries[event.Status.ID]
			select {
			case <-current.done:
			default:
				t.Errorf("terminal %s was published before its waiters were released", event.Status.State)
			}
			if current.cancel != nil {
				t.Errorf("terminal %s was published before its cancellation was spent", event.Status.State)
			}
			terminals <- event.Status
		})
		recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "atomic terminal recipe")
		runID := testutil.ArtifactID(t, artifact.KindRun, "atomic terminal run")
		request := Request{Task: recipe.TaskTraining, Recipe: recipeID}
		interrupt := make(chan struct{})
		id, err := manager.Submit(t.Context(), request, func(context.Context, Reporter) (Completion, error) {
			<-interrupt
			return Completion{Run: runID}, errors.New("interrupted")
		})
		if err != nil {
			t.Fatal(err)
		}
		first := waitInBackground(manager, id)
		synctest.Wait()
		close(interrupt)
		if status := <-terminals; status.State != StateFailed {
			t.Fatalf("first terminal status = %+v", status)
		}
		finish := make(chan struct{})
		if _, err := manager.Recover(t.Context(), id, request, func(ctx context.Context, _ Reporter) (Completion, error) {
			<-finish
			return Completion{Run: runID}, ctx.Err()
		}); err != nil {
			t.Fatal(err)
		}
		second := waitInBackground(manager, id)
		synctest.Wait()
		if outcome := <-first; outcome.err != nil || outcome.status.State != StateFailed {
			t.Fatalf("the first attempt's waiter = (%+v, %v)", outcome.status, outcome.err)
		}
		if status, _ := manager.Status(id); status.State != StateRunning {
			t.Fatalf("the recovery is %s before it finished", status.State)
		}
		close(finish)
		if outcome := <-second; outcome.err != nil || outcome.status.State != StateCompleted {
			t.Fatalf("the recovery's waiter = (%+v, %v)", outcome.status, outcome.err)
		}
		if status := <-terminals; status.State != StateCompleted || len(terminals) != 0 {
			t.Fatalf("recovered terminal status = %+v with %d more", status, len(terminals))
		}

		retaining, err := NewManager(1)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(retaining.Close)
		retained := make(chan Status, 2)
		retaining.Notify(func(event Event) {
			if terminal(event.Status.State) {
				retained <- event.Status
			}
		})
		complete := make(chan struct{})
		evicted, err := retaining.Submit(t.Context(), request, func(context.Context, Reporter) (Completion, error) {
			<-complete
			return Completion{Run: runID}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		attached := waitInBackground(retaining, evicted)
		synctest.Wait()
		close(complete)
		<-retained
		if _, err := retaining.Submit(t.Context(), request, func(context.Context, Reporter) (Completion, error) {
			return Completion{Run: runID}, nil
		}); err != nil {
			t.Fatal(err)
		}
		if _, present := retaining.Status(evicted); present {
			t.Fatal("retention kept a terminal entry past its limit")
		}
		if outcome := <-attached; outcome.err != nil || outcome.status.ID != evicted || outcome.status.State != StateCompleted {
			t.Fatalf("the evicted attempt's waiter = (%+v, %v)", outcome.status, outcome.err)
		}
	})
}

type waitOutcome struct {
	status Status
	err    error
}

// waitInBackground attaches a waiter to id and delivers what it returns.
func waitInBackground(manager *Manager, id artifact.ID) <-chan waitOutcome {
	outcome := make(chan waitOutcome, 1)
	go func() {
		status, err := manager.Wait(context.Background(), id)
		outcome <- waitOutcome{status: status, err: err}
	}()
	return outcome
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
				return Completion{Run: runID}, testutil.Recoverable(errors.New("approval required"), operatoraction.Block{
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
