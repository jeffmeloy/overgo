package server

import (
	"net/http"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
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
	Operation artifact.ID             `json:"operation"`
	Task      recipe.Task             `json:"task"`
	Recipe    artifact.ID             `json:"recipe"`
	Reason    string                  `json:"reason"`
	Subject   artifact.ID             `json:"subject"`
	Actions   []operatorBlockedAction `json:"actions,omitempty"`
	Evidence  []artifact.ID           `json:"evidence,omitempty"`
}

// operatorBlockedAction advertises one recovery action beside the exact
// approval-request identity a decision for it must name. The identity is
// content-derived from the operation, recipe, action, and decision chain,
// so a grant recorded against it can only bind the advertised facts.
type operatorBlockedAction struct {
	Code    string      `json:"code"`
	Request artifact.ID `json:"request"`
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
		var prior artifact.ID
		if decision, found, err := runrecord.ResolveHumanDecision(request.Context(), store, status.ID); err != nil {
			writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
			return
		} else if found {
			prior = decision.ID
		}
		for _, action := range status.Recovery.Actions {
			advertised, err := operatoraction.NewApprovalRequest(status.ID, status.Recipe, action, prior)
			if err != nil {
				writeError(response, http.StatusInternalServerError, "approval_derivation_failed", err.Error())
				return
			}
			blocked.Actions = append(blocked.Actions, operatorBlockedAction{Code: action.Code, Request: advertised.ID})
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
