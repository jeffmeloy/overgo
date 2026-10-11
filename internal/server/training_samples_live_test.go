package server

import (
	"encoding/json"
	"net/http"
	"slices"
	"testing"

	"overgo/internal/operation"
)

// liveStrip is one sample strip as the Train tab reads it off a running
// operation's text; the names are the ones the page reads.
type liveStrip struct {
	Record   string `json:"record"`
	Prompt   string `json:"prompt"`
	Expected string `json:"expected"`
	Samples  []struct {
		Step int    `json:"step"`
		Text string `json:"text"`
	} `json:"samples"`
}

// TestTrainingSamplesReachTheEventHub: a fine-tune of four updates sampled
// every second, started from the workbench route, carries its sample strips on
// the event hub while it runs, so the page need not wait for the verdict. The
// round taken before training arrives on an event of the running operation
// that has finished no update, and the next two on events that have finished
// the second and the fourth; each is the strips so far, under the names the
// page reads. The finished operation carries no text: the samples are then the
// third of its outputs.
func TestTrainingSamplesReachTheEventHub(t *testing.T) {
	t.Parallel()
	fixture := newSFTFixture(t)
	capabilities, err := fixture.workspace.WorkflowCapabilities(t.Context(), WorkflowTraining)
	if err != nil || len(capabilities) == 0 {
		t.Fatalf("training capabilities = %+v, %v", capabilities, err)
	}
	handler, err := New(Config{Repository: fixture.store, Workflows: fixture.workspace}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	events, unsubscribe, err := handler.events.subscribe(1024)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribe()
	body, err := json.Marshal(map[string]any{"task": "training", "recipe": capabilities[0].Recipe, "input": map[string]any{
		"dataset": fixture.conversations, "steps": 4, "sample_every": 2, "output": "live-samples",
	}})
	if err != nil {
		t.Fatal(err)
	}
	if started := serveTestRequest(handler, http.MethodPost, "/training/run", string(body)); started.Code != http.StatusAccepted {
		t.Fatalf("training run status=%d body=%s", started.Code, started.Body.String())
	}
	// Each text is kept once, as it first arrived, with the updates the run had
	// then finished.
	var texts []string
	var finished []uint64
	var final operation.Status
	for !slices.Contains(settledStates, final.State) {
		select {
		case event := <-events:
			update, ok := event.value.(operation.Event)
			if !ok {
				continue
			}
			final = update.Status
			if final.Text == "" || len(texts) != 0 && texts[len(texts)-1] == final.Text {
				continue
			}
			if final.State != operation.StateRunning {
				t.Fatalf("a text arrived on an event in state %q, want the running operation's", final.State)
			}
			texts, finished = append(texts, final.Text), append(finished, final.Progress.Completed)
		case <-t.Context().Done():
			t.Fatalf("the run did not finish on the hub: %v", t.Context().Err())
		}
	}
	if final.State != operation.StateCompleted || final.Text != "" || len(final.Outputs) != 3 {
		t.Fatalf("the run ended %q (%s) with a text of %d bytes and outputs %v, want it completed with no text and the trained model, the verdict and the samples",
			final.State, final.Failure, len(final.Text), final.Outputs)
	}
	if !slices.Equal(finished, []uint64{0, 2, 4}) {
		t.Fatalf("%d rounds reached the hub, after %v updates, want one before training, one after the second update and one after the fourth", len(texts), finished)
	}
	for round, text := range texts {
		var strips []liveStrip
		if err := json.Unmarshal([]byte(text), &strips); err != nil || len(strips) == 0 {
			t.Fatalf("round %d carried %q (%v), want the run's strips as JSON", round, text, err)
		}
		for _, strip := range strips {
			var steps []int
			for _, sample := range strip.Samples {
				steps = append(steps, sample.Step)
				if sample.Text == "" {
					t.Errorf("round %d: step %d of %s generated no text", round, sample.Step, strip.Record)
				}
			}
			if strip.Record == "" || strip.Prompt == "" || strip.Expected == "" || !slices.Equal(steps, []int{0, 2, 4}[:round+1]) {
				t.Errorf("round %d carried strip %+v, want its record, prompt and reply and the steps %v", round, strip, []int{0, 2, 4}[:round+1])
			}
		}
		t.Logf("round %d: %d strips in %d bytes of text on each event", round, len(strips), len(text))
	}
}
