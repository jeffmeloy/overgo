package operation

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestOperationRuntimeLifecycle(t *testing.T) {
	manager := newTestManager(t)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "operation-lifecycle-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "operation-lifecycle-run")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "operation-lifecycle-output")
	started, release := make(chan struct{}), make(chan struct{})
	id, err := manager.Submit(context.Background(), Request{Task: recipe.TaskGeneration, Recipe: recipeID},
		func(_ context.Context, reporter Reporter) (Completion, error) {
			total := uint64(2)
			reporter.Progress(1, &total)
			reporter.Metric(Metric{Name: "steps", Value: 1, Unit: "count"})
			close(started)
			<-release
			reporter.Publishing()
			return Completion{Run: runID, Outputs: []artifact.ID{outputID}}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	status, ok := manager.Status(id)
	if !ok || status.State != StateRunning || status.Progress.Completed != 1 || len(status.Metrics) != 1 {
		t.Fatalf("running status = %+v, present=%t", status, ok)
	}
	close(release)
	status, err = manager.Wait(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if status.State != StateCompleted || status.Run == nil || *status.Run != runID || len(status.Outputs) != 1 || status.Outputs[0] != outputID {
		t.Fatalf("completed status = %+v", status)
	}
}

func TestReporterMetricsRemainBounded(t *testing.T) {
	manager := newTestManager(t)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "bounded-metrics-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "bounded-metrics-run")
	id, err := manager.Submit(context.Background(), Request{Task: recipe.TaskTraining, Recipe: recipeID},
		func(_ context.Context, reporter Reporter) (Completion, error) {
			for value := range 100 {
				reporter.Metric(Metric{Name: "loss", Value: float64(value)})
			}
			return Completion{Run: runID}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if len(status.Metrics) != 1 || status.Metrics[0].Name != "loss" || status.Metrics[0].Value != 99 {
		t.Fatalf("metrics=%+v", status.Metrics)
	}
}

func TestServingAttemptEvidenceRemainsOrderedAndUnique(t *testing.T) {
	manager := newTestManager(t)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "attempt-evidence-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "attempt-evidence-run")
	first := testutil.ArtifactID(t, artifact.KindEvidence, "attempt-evidence-first")
	second := testutil.ArtifactID(t, artifact.KindEvidence, "attempt-evidence-second")
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskInference, Recipe: recipeID},
		func(_ context.Context, reporter Reporter) (Completion, error) {
			reporter.Attempt(first)
			reporter.Attempt(first)
			reporter.Attempt(runID)
			reporter.Attempt(second)
			return Completion{Run: runID}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(t.Context(), id)
	if err != nil || !slices.Equal(status.Attempts, []artifact.ID{first, second}) {
		t.Fatalf("attempt evidence = (%v, %v)", status.Attempts, err)
	}
}

func TestOperationEventStreamKeepsLatestBoundedState(t *testing.T) {
	manager := newTestManager(t)
	events, unsubscribe, err := manager.Subscribe()
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribe()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "operation event recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "operation event run")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "operation event output")
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskGeneration, Recipe: recipeID},
		func(_ context.Context, reporter Reporter) (Completion, error) {
			reporter.Attempt(testutil.ArtifactID(t, artifact.KindEvidence, "operation event attempt"))
			reporter.Metric(Metric{Name: "loss", Value: 1})
			reporter.Publishing()
			return Completion{Run: runID, Outputs: []artifact.ID{outputID}}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Wait(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	if cap(events) != operationEventBuffer || len(events) != operationEventBuffer {
		t.Fatalf("event queue capacity=%d length=%d", cap(events), len(events))
	}
	event := <-events
	if event.Sequence == 0 || event.Status.State != StateCompleted || event.Status.Run == nil ||
		*event.Status.Run != runID || !slices.Equal(event.Status.Outputs, []artifact.ID{outputID}) || len(event.Status.Attempts) != 1 {
		t.Fatalf("latest event = %+v", event)
	}
}

func TestOperationCancellationReleasesResources(t *testing.T) {
	manager := newTestManager(t)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "operation-cancel-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "operation-cancel-run")
	started, released := make(chan struct{}), make(chan struct{})
	id, err := manager.Submit(context.Background(), Request{Task: recipe.TaskTraining, Recipe: recipeID},
		func(ctx context.Context, _ Reporter) (Completion, error) {
			close(started)
			defer close(released)
			<-ctx.Done()
			return Completion{Run: runID}, ctx.Err()
		})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	if !manager.Cancel(id) {
		t.Fatal("active operation was not cancelled")
	}
	status, err := manager.Wait(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	<-released
	if status.State != StateCancelled || status.Run == nil || *status.Run != runID {
		t.Fatalf("cancelled status = %+v", status)
	}
}

func TestOperationPublicationIsAtomic(t *testing.T) {
	manager := newTestManager(t)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "operation-publication-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "operation-publication-run")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "operation-publication-output")
	publishing, commit := make(chan struct{}), make(chan struct{})
	id, err := manager.Submit(context.Background(), Request{Task: recipe.TaskGeneration, Recipe: recipeID},
		func(_ context.Context, reporter Reporter) (Completion, error) {
			reporter.Publishing()
			close(publishing)
			<-commit
			return Completion{Run: runID, Outputs: []artifact.ID{outputID}}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	<-publishing
	status, ok := manager.Status(id)
	if !ok || status.State != StatePublishing || status.Run != nil || len(status.Outputs) != 0 {
		t.Fatalf("pre-commit status = %+v, present=%t", status, ok)
	}
	close(commit)
	status, err = manager.Wait(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if status.State != StateCompleted || status.Run == nil || *status.Run != runID || len(status.Outputs) != 1 {
		t.Fatalf("published status = %+v", status)
	}
}

func newTestManager(t *testing.T) *Manager {
	t.Helper()
	manager, err := NewManager(8)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(manager.Close)
	return manager
}
