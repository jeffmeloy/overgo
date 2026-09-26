package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/operation"
)

// TestTrainingStepsReachTheEventHub: a training run started from the
// workbench route reports each step on the event hub the page subscribes to,
// in order against the run's total, with the step loss a trend the page can
// redraw after a reload.
func TestTrainingStepsReachTheEventHub(t *testing.T) {
	t.Parallel()
	fixture := newDPOTrainingFixture(t)
	handler, err := New(Config{Repository: fixture.store}, &workspaceTestRuntime{fakeGenerator: &fakeGenerator{}, WorkflowWorkspaceAPI: fixture.workspace})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	events, unsubscribe, err := handler.events.subscribe(1024)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribe()
	const steps = 3
	body, err := json.Marshal(map[string]any{"task": "training", "recipe": fixture.definition.ID, "input": map[string]any{
		"dataset": fixture.dataset, "output": "live", "steps": steps, "objective_scale": 0.1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if started := serveTestRequest(handler, http.MethodPost, "/training/run", string(body)); started.Code != http.StatusAccepted {
		t.Fatalf("training run status=%d body=%s", started.Code, started.Body.String())
	}
	var reported []uint64
	var final operation.Status
	for final.State != operation.StateCompleted {
		select {
		case event := <-events:
			update, ok := event.value.(operation.Event)
			if !ok {
				continue
			}
			status := update.Status
			if status.State == operation.StateFailed {
				t.Fatalf("training failed: %s", status.Failure)
			}
			if total := status.Progress.Total; total != nil && *total == steps && !endsAt(reported, status.Progress.Completed) {
				reported = append(reported, status.Progress.Completed)
			}
			final = status
		case <-t.Context().Done():
			t.Fatalf("the run did not finish on the hub: %v", t.Context().Err())
		}
	}
	if len(reported) != steps+1 || reported[0] != 0 || reported[steps] != steps {
		t.Fatalf("progress on the hub = %v, want 0..%d in order", reported, steps)
	}
	var loss *operation.Series
	for index := range final.Series {
		if final.Series[index].Name == "dpo_loss" {
			loss = &final.Series[index]
		}
	}
	if loss == nil || loss.Reports != steps || len(loss.Values) != steps {
		t.Fatalf("loss trend = %+v in %+v", loss, final.Series)
	}
}

// endsAt: the progress already ends at this count.
func endsAt(reported []uint64, completed uint64) bool {
	return len(reported) != 0 && reported[len(reported)-1] == completed
}
