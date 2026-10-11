package operation

import (
	"context"
	"slices"
	"strings"
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
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskGeneration, Recipe: recipeID},
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
	status, err = manager.Wait(t.Context(), id)
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
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskTraining, Recipe: recipeID},
		func(_ context.Context, reporter Reporter) (Completion, error) {
			for value := range 100 {
				reporter.Metric(Metric{Name: "loss", Value: float64(value)})
			}
			return Completion{Run: runID}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	if len(status.Metrics) != 1 || status.Metrics[0].Name != "loss" || status.Metrics[0].Value != 99 {
		t.Fatalf("metrics=%+v", status.Metrics)
	}
	// The repeated metric is a trend: every report counted, the values
	// within the manager's bound, value i the report i*stride.
	if len(status.Series) != 1 || status.Series[0].Name != "loss" || status.Series[0].Reports != 100 {
		t.Fatalf("series=%+v", status.Series)
	}
	trend := status.Series[0]
	if len(trend.Values) > 8 || trend.Stride < 2 {
		t.Fatalf("trend is not bounded: %+v", trend)
	}
	for index, value := range trend.Values {
		if value != float64(uint64(index)*trend.Stride) {
			t.Fatalf("trend value %d = %v, want report %d", index, value, uint64(index)*trend.Stride)
		}
	}
}

// TestReporterTextStaysBounded holds the one text a running operation carries.
// A later report replaces the earlier one and reaches the observer on an event
// of the running operation; a text of exactly the bound is taken and one a byte
// past it is refused whole, changing nothing and publishing nothing, because a
// cut text would no longer parse; and once the operation has finished, its
// status and its terminal event carry no text, and a late report is ignored.
func TestReporterTextStaysBounded(t *testing.T) {
	manager := newTestManager(t)
	var events []Event
	manager.Notify(func(event Event) { events = append(events, event) })
	// The observer appends under the manager's lock, so a read takes it too.
	seen := func() []Event {
		manager.mu.RLock()
		defer manager.mu.RUnlock()
		return slices.Clone(events)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "bounded-text-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "bounded-text-run")
	atBound := strings.Repeat("s", textBytesMost)
	started, release := make(chan struct{}), make(chan struct{})
	var late Reporter
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskTraining, Recipe: recipeID},
		func(ctx context.Context, reporter Reporter) (Completion, error) {
			reporter.Text("first")
			reporter.Text(atBound)
			reporter.Text(atBound + "x")
			late = reporter
			close(started)
			// A check that fails before the release ends the run through the
			// manager's Close, which cancels it, not by waiting on it for ever.
			select {
			case <-release:
			case <-ctx.Done():
			}
			return Completion{Run: runID}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	status, ok := manager.Status(id)
	if !ok || status.State != StateRunning || status.Text != atBound {
		t.Fatalf("running status holds a text of %d bytes in state %q (present=%t), want the second report, %d bytes, the one past the bound refused",
			len(status.Text), status.State, ok, len(atBound))
	}
	// Each accepted report is one event of the running operation, in order, and
	// the refused one is none: the last event is the second report's.
	var texts []string
	for _, event := range seen() {
		if event.Status.Text == "" {
			continue
		}
		if event.Status.State != StateRunning {
			t.Fatalf("a text arrived on an event in state %q, want running", event.Status.State)
		}
		texts = append(texts, event.Status.Text)
	}
	if !slices.Equal(texts, []string{"first", atBound}) {
		t.Fatalf("the observer saw %d events with a text, of %d bytes each in turn, want the first report then the second and no other", len(texts), lengths(texts))
	}
	running := len(seen())
	close(release)
	status, err = manager.Wait(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	if status.State != StateCompleted || status.Text != "" {
		t.Fatalf("finished status is %q with a text of %d bytes, want completed and no text", status.State, len(status.Text))
	}
	after := seen()
	if terminalEvent := after[len(after)-1]; len(after) != running+1 || terminalEvent.Status.State != StateCompleted || terminalEvent.Status.Text != "" {
		t.Fatalf("the run ended with %d more events, the last %q with a text of %d bytes, want one terminal event without text",
			len(after)-running, terminalEvent.Status.State, len(terminalEvent.Status.Text))
	}
	// A report for an attempt that has ended is ignored, as its other reports are.
	late.Text("late")
	if status, _ := manager.Status(id); status.Text != "" || len(seen()) != len(after) {
		t.Fatalf("a report after the end left the text %q and %d more events, want it ignored", status.Text, len(seen())-len(after))
	}
}

// lengths is each text's length, for a failure message that would otherwise
// print a text as long as the bound.
func lengths(texts []string) []int {
	result := make([]int, len(texts))
	for index, text := range texts {
		result[index] = len(text)
	}
	return result
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

func TestOperationTransitionsReachTheObserverInOrder(t *testing.T) {
	manager := newTestManager(t)
	var events []Event
	manager.Notify(func(event Event) { events = append(events, event) })
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
	// Every transition arrives, each after the one before it, ending in the terminal one.
	manager.mu.Lock()
	defer manager.mu.Unlock()
	for index := range events {
		if events[index].Sequence != uint64(index+1) {
			t.Fatalf("transition %d carries sequence %d", index, events[index].Sequence)
		}
	}
	if len(events) < 3 {
		t.Fatalf("observer saw %d transitions", len(events))
	}
	event := events[len(events)-1]
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
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskTraining, Recipe: recipeID},
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
	status, err := manager.Wait(t.Context(), id)
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
	id, err := manager.Submit(t.Context(), Request{Task: recipe.TaskGeneration, Recipe: recipeID},
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
	status, err = manager.Wait(t.Context(), id)
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
