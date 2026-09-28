package server

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"testing"
)

// TestTurnPhaseProfile: a turn's tokens per forward counts each accepted
// draft as a saved forward; the inspector and the runtime snapshot carry the
// recent finished turns' decode rate and tokens per forward oldest first;
// and at its bound the registry lets the oldest finished turn go first.
func TestTurnPhaseProfile(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		generated, accepted uint64
		want                float64
	}{{10, 0, 1}, {10, 5, 2}, {9, 6, 3}, {0, 0, 0}} {
		if got := tokensPerForward(tc.generated, tc.accepted); got != tc.want {
			t.Errorf("tokensPerForward(%d, %d) = %v, want %v", tc.generated, tc.accepted, got, tc.want)
		}
	}

	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"one ", "two"}}))
	// The workbench streams its stored turns; the server holds each one's final with its timings.
	var responses []string
	for range 3 {
		response := serveTestRequest(handler, http.MethodPost, "/v1/responses", `{"model":"`+testModelID+`","input":"profile","store":true,"stream":true,"max_output_tokens":4}`)
		_, created, found := strings.Cut(response.Body.String(), "event: response.created\ndata: ")
		var event struct {
			Response struct {
				ID string `json:"id"`
			} `json:"response"`
		}
		if line, _, _ := strings.Cut(created, "\n"); !found || json.Unmarshal([]byte(line), &event) != nil || event.Response.ID == "" {
			t.Fatalf("turn %d %s", response.Code, response.Body.String())
		}
		responses = append(responses, event.Response.ID)
	}
	var inspected struct {
		Timings struct {
			TokensPerForward float64 `json:"tokens_per_forward"`
		} `json:"timings"`
		Recent []turnProfile `json:"recent"`
	}
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/interactions/inspect?response="+responses[2], "").Body.Bytes(), &inspected); err != nil {
		t.Fatal(err)
	}
	profiled := func(recent []turnProfile) []string {
		ids := []string{}
		for _, turn := range recent {
			if turn.TokensPerForward != 1 || turn.PredictedPerSecond <= 0 {
				t.Fatalf("recent turn %+v", turn)
			}
			ids = append(ids, turn.Response)
		}
		return ids
	}
	if inspected.Timings.TokensPerForward != 1 || !slices.Equal(profiled(inspected.Recent), responses) {
		t.Fatalf("inspected %+v, want the turns %v", inspected, responses)
	}
	var sessions runtimeSessionsResponse
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/runtime/sessions", "").Body.Bytes(), &sessions); err != nil || !slices.Equal(profiled(sessions.Recent), responses) {
		t.Fatalf("runtime recent %+v, %v", sessions.Recent, err)
	}

	var registry inflightRegistry
	for _, id := range []string{"first", "second"} {
		registry.begin(id, 2, func() {}).finish(responsesResponse{ID: id, Status: "completed"}, "")
	}
	if registry.begin("third", 2, func() {}) == nil {
		t.Fatal("a full registry of finished turns refused a new one")
	}
	if _, kept := registry.lookup("first"); kept {
		t.Fatal("the registry kept the oldest finished turn and let a newer one go")
	}
	if _, kept := registry.lookup("second"); !kept {
		t.Fatal("the registry let the newer finished turn go")
	}
}
