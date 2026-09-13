package server

import (
	"context"
	"net/http"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/runrecord"
)

// Inspect this turn (professional GUI campaign, gui-workbench/inspect-turn):
// the front page opens a side panel over any assistant turn with the
// turn's run record and the analysis inspectors seeded with its exact
// prompt and completion. The record's status comes from one declared
// vocabulary: derived from the in-flight registry while the turn runs or
// while its final is still held, and from the durable interaction and its
// trace once recorded.

// The turn status vocabulary.
const (
	turnStatusRunning   = "running"   // still generating; the in-flight registry holds it
	turnStatusDone      = "done"      // recorded, or finished with its final held
	turnStatusCancelled = "cancelled" // the client or the server stopped it
	turnStatusTimeout   = "timeout"   // the request deadline ended it
	turnStatusRefused   = "refused"   // admission refused it before generation
	turnStatusError     = "error"     // generation failed
)

var turnStatuses = []string{turnStatusRunning, turnStatusDone, turnStatusCancelled, turnStatusTimeout, turnStatusRefused, turnStatusError}

// turnInspection is the run record of one turn as the side panel shows it.
type turnInspection struct {
	Response   string             `json:"response"`
	Status     string             `json:"status"`
	Statuses   []string           `json:"statuses"`
	Failure    string             `json:"failure,omitzero"`
	Timings    *slotStatusTimings `json:"timings,omitempty"`
	Sampling   *responseSampling  `json:"sampling,omitempty"`
	Model      artifact.ID        `json:"model,omitzero"`
	Recipe     artifact.ID        `json:"recipe,omitzero"`
	Trace      artifact.ID        `json:"trace,omitzero"`
	Receipt    artifact.ID        `json:"receipt,omitzero"`
	Operation  artifact.ID        `json:"operation,omitzero"`
	Run        artifact.ID        `json:"run,omitzero"`
	Prompt     string             `json:"prompt"`
	Completion string             `json:"completion"`
	// Reproducible is false for a turn served at a remote provider: its
	// environment records the remote backend, and nothing about the
	// hosted model can be replayed from the store.
	Reproducible bool `json:"reproducible"`
}

// turnFailureStatus reads a failed turn's message into the vocabulary.
func turnFailureStatus(failure string) string {
	lower := strings.ToLower(failure)
	switch {
	case strings.Contains(lower, "cancel"):
		return turnStatusCancelled
	case strings.Contains(lower, "timeout") || strings.Contains(lower, "deadline"):
		return turnStatusTimeout
	case strings.Contains(lower, "refus"):
		return turnStatusRefused
	}
	return turnStatusError
}

// conversationInspect answers GET /interactions/inspect?response= with the
// turn's run record.
func (h *Handler) conversationInspect(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	responseID := request.URL.Query().Get("response")
	result := turnInspection{Response: responseID, Statuses: turnStatuses, Reproducible: h.environment.Reproducible()}
	if turn, found := h.inflight.lookup(responseID); found {
		text, done, final, failed, _ := turn.snapshot()
		result.Completion, result.Status = text, turnStatusRunning
		if done && failed != "" {
			result.Status, result.Failure = turnFailureStatus(failed), failed
		} else if done {
			result.Status = turnStatusDone
		}
		result.Timings, result.Sampling = final.Timings, final.Sampling
	}
	if messages, _, found := h.loadResponseInteraction(request.Context(), responseID); found {
		interaction, _, _ := runrecord.ResolveInteraction(request.Context(), h.repository, responseID)
		if result.Status == "" {
			status, failure := h.responseTerminal(request.Context(), interaction)
			result.Status = turnStatusDone
			if status == "cancelled" {
				result.Status = turnStatusCancelled
			} else if status != "completed" {
				result.Status = turnStatusError
			}
			result.Failure = failure
		}
		result.Model, result.Recipe, result.Trace = interaction.Model, interaction.Recipe, interaction.Trace
		result.Operation, result.Run = interaction.Operation, interaction.Run
		if trace, err := runrecord.RequireInteractionTrace(request.Context(), h.repository, interaction.Trace); err == nil {
			result.Receipt = trace.Request
		}
		for _, message := range messages {
			switch message.Role {
			case inference.ChatRoleUser:
				result.Prompt = message.Content
			case inference.ChatRoleAssistant:
				result.Completion = message.Content
			}
		}
	}
	if result.Status == "" {
		writeError(response, http.StatusNotFound, "not_found", "turn not found")
		return
	}
	writeJSON(response, http.StatusOK, result)
}

// conversationMessage is one visible message with the response that
// produced it, so a page can inspect the turn behind any assistant message.
type conversationMessage struct {
	Role      string                          `json:"role"`
	Content   string                          `json:"content"`
	Response  string                          `json:"response,omitzero"`
	Media     []runrecord.InteractionMedia    `json:"media,omitempty"`
	ToolCalls []runrecord.InteractionToolCall `json:"tool_calls,omitempty"`
}

// chainMessages materializes a chain root-first, attributing each message
// to the response whose interaction introduced it.
func (h *Handler) chainMessages(ctx context.Context, chain []runrecord.Interaction) ([]conversationMessage, error) {
	result := []conversationMessage{}
	for index := len(chain) - 1; index >= 0; index-- {
		interaction := chain[index]
		transcript, err := runrecord.RequireInteractionTranscript(ctx, h.repository, interaction.Message)
		if err != nil {
			return nil, err
		}
		for _, message := range transcript.Messages {
			result = append(result, conversationMessage{Role: message.Role, Content: message.Content, Response: interaction.Response, Media: message.Media, ToolCalls: message.ToolCalls})
		}
	}
	return result, nil
}
