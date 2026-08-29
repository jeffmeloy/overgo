package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/runrecord"
)

func TestReplayGUI(t *testing.T) {
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	handler.publishResponseInteraction(t.Context(), "resp_trace", artifact.ID{}, []inference.ChatMessage{
		{Role: inference.ChatRoleUser, Content: "question"},
		{Role: inference.ChatRoleAssistant, Content: "answer"},
	})
	response := serveTestRequest(handler, http.MethodGet, "/interactions/replay?response=resp_trace", "")
	if response.Code != http.StatusOK {
		t.Fatalf("replay status=%d body=%s", response.Code, response.Body.String())
	}
	var replay interactionReplayResponse
	if err := json.Unmarshal(response.Body.Bytes(), &replay); err != nil {
		t.Fatal(err)
	}
	if replay.ID.Kind() != artifact.KindEvidence || replay.Trace.Request.Kind() != artifact.KindEvidence ||
		len(replay.Trace.Events) != 2 || replay.Trace.Events[1].Kind != runrecord.InteractionEventOutput {
		t.Fatalf("replay response = %+v", replay)
	}
}
