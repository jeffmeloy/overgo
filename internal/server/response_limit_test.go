package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Exercise the wire result and its durable projection with the same generation.
func TestResponsesIncompleteAtOutputLimit(t *testing.T) {
	requestMessages := []map[string]string{
		{"role": "user", "content": "previous question"},
		{"role": "assistant", "content": "previous complete answer"},
		{"role": "user", "content": "bounded answer"},
	}
	for _, tc := range []struct {
		name, stop, status, text string
		limit                    int
	}{
		{"limit", "", "incomplete", "A", 1},
		{"natural", "", "completed", "AB", 3},
		{"stop_at_limit", "B", "completed", "A", 2},
	} {
		for _, stream := range []bool{false, true} {
			for _, stored := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%t/store=%t", tc.name, stream, stored), func(t *testing.T) {
					store, err := overgodb.Open(t.TempDir())
					if err != nil {
						t.Fatal(err)
					}
					defer store.Close()
					handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
					input := map[string]any{"input": requestMessages, "max_output_tokens": tc.limit, "stream": stream, "store": stored}
					if tc.stop != "" {
						input["stop"] = tc.stop
					}
					body, err := json.Marshal(input)
					if err != nil {
						t.Fatal(err)
					}
					response := serveTestRequest(handler, http.MethodPost, "/v1/responses", string(body))
					if response.Code != http.StatusOK {
						t.Fatalf("response: %d %s", response.Code, response.Body)
					}
					var final responsesResponse
					if stream {
						var terminals int
						for frame := range strings.SplitSeq(response.Body.String(), "\n\n") {
							if !strings.HasPrefix(frame, "event: response."+tc.status+"\n") {
								continue
							}
							_, data, ok := strings.Cut(frame, "\ndata: ")
							var event struct {
								Response responsesResponse `json:"response"`
							}
							if !ok || json.Unmarshal([]byte(data), &event) != nil {
								t.Fatalf("terminal frame: %s", frame)
							}
							final = event.Response
							terminals++
						}
						if terminals != 1 {
							t.Fatalf("terminal count=%d: %s", terminals, response.Body)
						}
						if tc.status == "incomplete" && strings.Contains(response.Body.String(), "event: response.completed") {
							t.Fatal("limit also reported completed")
						}
					} else if err := json.Unmarshal(response.Body.Bytes(), &final); err != nil {
						t.Fatal(err)
					}
					if final.ID == "" || final.Status != tc.status || len(final.Output) != 1 || final.Output[0].Content[0].Text != tc.text {
						t.Fatalf("final = %+v", final)
					}
					checkDetails := func(details *responseIncompleteDetails) {
						t.Helper()
						if tc.status == "incomplete" {
							if details == nil || details.Reason != "max_output_tokens" {
								t.Fatalf("incomplete details = %+v", details)
							}
						} else if details != nil {
							t.Fatalf("natural stop has incomplete details: %+v", details)
						}
					}
					checkDetails(final.IncompleteDetails)
					if final.Output[0].Status != tc.status || (final.CompletedAt == nil) != (tc.status == "incomplete") {
						t.Fatalf("output or completion timestamp contradicts terminal status: %+v", final)
					}
					interaction, found, err := runrecord.ResolveInteraction(t.Context(), store, final.ID)
					if err != nil || found != stored {
						t.Fatalf("stored=%t found=%t err=%v", stored, found, err)
					}
					if stored {
						if (interaction.TerminalReason == runrecord.InteractionOutputLimit) != (tc.status == "incomplete") {
							t.Fatalf("stored reason = %s", interaction.TerminalReason)
						}
						checkHistory := func(h *Handler) {
							t.Helper()
							follow := serveTestRequest(h, http.MethodGet, "/interactions/follow?response="+final.ID, "")
							if follow.Code != http.StatusOK || !strings.Contains(follow.Body.String(), "event: response."+tc.status) || strings.Count(follow.Body.String(), `"delta":"`+tc.text+`"`) != 1 {
								t.Fatalf("follow: %d %s", follow.Code, follow.Body)
							}
							if tc.status == "incomplete" && (!strings.Contains(follow.Body.String(), `"reason":"max_output_tokens"`) || strings.Contains(follow.Body.String(), "event: response.completed")) {
								t.Fatalf("limit replay: %s", follow.Body)
							}
							messages := serveTestRequest(h, http.MethodGet, "/interactions/messages?response="+final.ID, "")
							var history conversationMessagesResponse
							if err := json.Unmarshal(messages.Body.Bytes(), &history); err != nil || history.Status != tc.status || history.Failure != "" {
								t.Fatalf("history: %s (%v)", messages.Body, err)
							}
							checkDetails(history.IncompleteDetails)
							inspect := serveTestRequest(h, http.MethodGet, "/interactions/inspect?response="+final.ID, "")
							var record turnInspection
							wantStatus := turnStatusDone
							if tc.status == "incomplete" {
								wantStatus = turnStatusIncomplete
							}
							if err := json.Unmarshal(inspect.Body.Bytes(), &record); err != nil || record.Status != wantStatus || record.Failure != "" {
								t.Fatalf("inspection: %s (%v)", inspect.Body, err)
							}
							checkDetails(record.IncompleteDetails)
							if len(history.Messages) != len(requestMessages)+1 || history.Messages[len(requestMessages)].Content != tc.text {
								t.Fatalf("messages = %+v", history.Messages)
							}
							for index, input := range requestMessages {
								message := history.Messages[index]
								if message.Role != input["role"] || message.Content != input["content"] || message.IncompleteDetails != nil {
									t.Fatalf("request message changed or inherited terminal reason: %+v", message)
								}
							}
							checkDetails(history.Messages[len(requestMessages)].IncompleteDetails)
						}
						checkHistory(handler)
						if err := handler.Close(); err != nil {
							t.Fatal(err)
						}
						restarted := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
						checkHistory(restarted)
						continuation := serveTestRequest(restarted, http.MethodPost, "/v1/responses", `{"input":"continue","previous_response_id":"`+final.ID+`","max_output_tokens":3}`)
						if continuation.Code != http.StatusOK {
							t.Fatalf("continuation: %s", continuation.Body)
						}
					}
				})
			}
		}
	}
}
