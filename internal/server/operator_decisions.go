package server

import (
	"net/http"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const operatorDecisionViewLimit = 200

// operatorDecisionView is the one pending-decision surface: every committed
// approval request awaiting a human decision beside every blocked operation
// holding typed recovery actions. Rows cite canonical evidence identities;
// the actions they expose are the existing idempotent decision and recovery
// endpoints — the view creates no new decision authority.
type operatorDecisionView struct {
	Approvals []runrecord.PendingOperatorDecision `json:"approvals"`
	Blocked   []operatorBlockedOperation          `json:"blocked"`
}

type operatorBlockedOperation struct {
	Operation artifact.ID   `json:"operation"`
	Task      recipe.Task   `json:"task"`
	Recipe    artifact.ID   `json:"recipe"`
	Reason    string        `json:"reason"`
	Subject   artifact.ID   `json:"subject"`
	Actions   []string      `json:"actions,omitempty"`
	Evidence  []artifact.ID `json:"evidence,omitempty"`
}

// operatorDecisions serves the single pending-decision view.
func (h *Handler) operatorDecisions(response http.ResponseWriter, request *http.Request) {
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	approvals, err := runrecord.PendingOperatorDecisions(request.Context(), store, operatorDecisionViewLimit)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	view := operatorDecisionView{Approvals: approvals, Blocked: []operatorBlockedOperation{}}
	for _, status := range h.operations.List() {
		if status.State != operation.StateBlocked || status.Recovery == nil {
			continue
		}
		blocked := operatorBlockedOperation{
			Operation: status.ID, Task: status.Task, Recipe: status.Recipe,
			Reason: status.Recovery.Reason, Subject: status.Recovery.Subject,
			Evidence: status.Recovery.Evidence,
		}
		for _, action := range status.Recovery.Actions {
			blocked.Actions = append(blocked.Actions, action.Code)
		}
		view.Blocked = append(view.Blocked, blocked)
	}
	writeJSON(response, http.StatusOK, view)
}

// operatorTimeline serves the one causal operation timeline.
func (h *Handler) operatorTimeline(response http.ResponseWriter, request *http.Request) {
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	id, err := artifact.ParseID(request.URL.Query().Get("id"))
	if err != nil || id.Kind() != artifact.KindEvidence {
		writeError(response, http.StatusBadRequest, "invalid_operation", "operation evidence identity is required")
		return
	}
	limit := operatorDecisionViewLimit
	if value := request.URL.Query().Get("limit"); value != "" {
		parsed, parseErr := strconv.Atoi(value)
		if parseErr != nil || parsed <= 0 || parsed > operatorDecisionViewLimit {
			writeInvalidRequestMessage(response, "invalid timeline bound")
			return
		}
		limit = parsed
	}
	events, err := runrecord.DeriveOperatorTimeline(request.Context(), store, id, limit)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"operation": id, "events": events})
}
