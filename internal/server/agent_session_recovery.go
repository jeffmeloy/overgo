package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// An agent session's durable entries sit under the response alias root as
// <agent>:<session>-turn-N (a chat turn) and <agent>:<session>-step-N (a
// tool step the coordinator recorded), so the session reads back in the
// order they were committed.
const (
	agentTurnEntry = "turn"
	agentStepEntry = "step"
)

// agentSessionID names an agent's session as its entries are keyed.
func agentSessionID(agent, session string) string {
	if agent == "" {
		return session
	}
	return agent + ":" + session
}

// bufferedResponse holds a response until it is inspected.
type bufferedResponse struct {
	header http.Header
	status int
	body   bytes.Buffer
}

// Header is the held response's header.
func (buffered *bufferedResponse) Header() http.Header { return buffered.header }

// Write holds the body.
func (buffered *bufferedResponse) Write(data []byte) (int, error) { return buffered.body.Write(data) }

// WriteHeader holds the status.
func (buffered *bufferedResponse) WriteHeader(status int) { buffered.status = status }

// relay writes the held response on.
func (buffered *bufferedResponse) relay(response http.ResponseWriter) {
	maps.Copy(response.Header(), buffered.header)
	response.WriteHeader(writtenStatus(buffered.status))
	_, _ = response.Write(buffered.body.Bytes())
}

// writtenStatus: a handler that wrote without a status wrote 200.
func writtenStatus(status int) int {
	if status == 0 {
		return http.StatusOK
	}
	return status
}

// recordAgentTurn stores one answered chat turn of a session as its next
// turn entry, its parent the session's previous turn.
func (h *Handler) recordAgentTurn(ctx context.Context, sessionID string, user, assistant inference.ChatMessage) error {
	h.agentSessions.turns.Lock()
	defer h.agentSessions.turns.Unlock()
	var parent artifact.ID
	for turn := 1; turn <= h.config.MaxStoredResponses; turn++ {
		responseID := sessionID + "-" + agentTurnEntry + "-" + strconv.Itoa(turn)
		previous, found, err := runrecord.ResolveInteraction(ctx, h.repository, responseID)
		if err != nil {
			return err
		}
		if !found {
			return h.publishResponseInteraction(ctx, responseID, parent, []inference.ChatMessage{user, assistant}, runrecord.OutcomeSucceeded, "", turnRecord{})
		}
		parent = previous.ID
	}
	return errors.New("server: the agent session holds its bound of turns")
}

// agentThreadEntry: one entry of a session's thread: a chat turn's user and
// assistant text, or a tool step's call and its result.
type agentThreadEntry struct {
	Kind        string          `json:"kind"`
	Call        string          `json:"call"`
	Interaction string          `json:"interaction"`
	User        string          `json:"user,omitzero"`
	Assistant   string          `json:"assistant,omitzero"`
	Tool        string          `json:"tool,omitzero"`
	Arguments   json.RawMessage `json:"arguments,omitempty"`
	Result      string          `json:"result,omitzero"`
	Error       bool            `json:"error,omitzero"`
}

type agentThreadResponse struct {
	Session   string             `json:"session"`
	Steps     int                `json:"steps"`
	Entries   []agentThreadEntry `json:"entries"`
	Truncated bool               `json:"truncated"`
}

// agentThread reads a session's turns and steps back in commit order, as a
// restarted server or a reloaded page resumes it.
func (h *Handler) agentThread(response http.ResponseWriter, request *http.Request) {
	values := request.URL.Query()
	if values.Get("session") == "" {
		writeInvalidRequestMessage(response, "session is required")
		return
	}
	sessionID := agentSessionID(values.Get("agent"), values.Get("session"))
	thread := agentThreadResponse{Session: sessionID, Entries: []agentThreadEntry{}}
	page, err := overgodb.VisitDecodedDocuments(request.Context(), h.repository, overgodb.DocumentQuery{
		Contracts:     []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.InteractionMediaType, Schema: runrecord.InteractionSchema}},
		AliasPrefixes: []string{runrecord.InteractionResponseAliasRoot + sessionID + "-"},
		Order:         overgodb.DocumentOldestFirst, MaxResults: h.config.MaxStoredResponses,
	}, runrecord.ParseInteraction, func(_ overgodb.DocumentView, interaction runrecord.Interaction) error {
		kind, number, named := strings.Cut(strings.TrimPrefix(interaction.Response, sessionID+"-"), "-")
		if _, err := strconv.Atoi(number); !named || err != nil || (kind != agentTurnEntry && kind != agentStepEntry) {
			return nil // another session whose name extends this one's
		}
		transcript, err := runrecord.RequireInteractionTranscript(request.Context(), h.repository, interaction.Message)
		if err != nil {
			return err
		}
		entry := agentThreadEntry{Kind: kind, Call: interaction.Response, Interaction: idText(interaction.ID)}
		for _, message := range transcript.Messages {
			switch {
			case message.ToolCallID != "":
				entry.Result, entry.Error = message.Content, message.ToolResultError
			case len(message.ToolCalls) != 0:
				entry.Tool = message.ToolCalls[0].Name
				if message.ToolCalls[0].Arguments != "" {
					entry.Arguments = json.RawMessage(message.ToolCalls[0].Arguments)
				}
			case message.Role == string(inference.ChatRoleUser):
				entry.User = message.Content
			case message.Role == string(inference.ChatRoleAssistant):
				entry.Assistant = message.Content
			}
		}
		if kind == agentStepEntry {
			thread.Steps++
		}
		thread.Entries = append(thread.Entries, entry)
		return nil
	})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	thread.Truncated = page.Truncated
	writeJSON(response, http.StatusOK, thread)
}
