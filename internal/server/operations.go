package server

import (
	"context"
	"errors"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

type operationCancelRequest struct {
	ID artifact.ID `json:"id"`
}

type operationDecisionRequest struct {
	Operation artifact.ID           `json:"operation"`
	Tool      string                `json:"tool"`
	Answer    operatoraction.Answer `json:"answer"`
}

type operationDecisionResponse struct {
	Decision  artifact.ID `json:"decision"`
	Operation artifact.ID `json:"operation"`
}

func (h *Handler) operationStatus(response http.ResponseWriter, request *http.Request) {
	value := request.URL.Query().Get("id")
	if value == "" {
		writeJSON(response, http.StatusOK, h.operations.List())
		return
	}
	id, err := artifact.ParseID(value)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	status, ok := h.operations.Status(id)
	if !ok {
		writeError(response, http.StatusNotFound, "not_found", "operation not found")
		return
	}
	writeJSON(response, http.StatusOK, status)
}

// operationDAG projects one operation's workflow as its recipe graph
// joined with the durable stage receipts: every node with its module
// and latest lifecycle state (pending when no receipt exists yet), and
// every edge of the definition -- so the live DAG view reads directly
// from what the runtime durably wrote, and a waiting node points the
// operator at the inbox.
func (h *Handler) operationDAG(response http.ResponseWriter, request *http.Request) {
	id, err := artifact.ParseID(request.URL.Query().Get("id"))
	if err != nil || id.Kind() != artifact.KindEvidence {
		writeError(response, http.StatusBadRequest, "invalid_operation", "operation evidence identity is required")
		return
	}
	projection, err := h.operationEvidenceSnapshot(request.Context(), id, h.config.MaxStoredResponses)
	if errors.Is(err, errBrowseRepositoryUnavailable) {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "operation evidence repository is not configured")
		return
	}
	if errors.Is(err, errOperationEvidenceNotFound) {
		writeError(response, http.StatusNotFound, "operation_not_found", err.Error())
		return
	}
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	states := map[recipe.NodeID]runrecord.StageReceipt{}
	var recipeID artifact.ID
	for _, document := range projection.Stages {
		states[document.Value.Node] = document.Value
		recipeID = document.Value.Recipe
	}
	if !recipeID.Valid() {
		if status, live := h.operations.Status(id); live {
			recipeID = status.Recipe
		}
	}
	if !recipeID.Valid() {
		writeError(response, http.StatusNotFound, "operation_not_found", "no stage receipts or live status name this operation's recipe")
		return
	}
	store, err := h.browseStore(request.Context())
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	content, found, err := artifact.ReadContent(request.Context(), store, recipeID)
	if err != nil || !found {
		writeError(response, http.StatusNotFound, "recipe_not_found", "the operation's recipe definition is not committed")
		return
	}
	definition, err := recipe.ParseDefinition(content.Data)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	nodes := make([]map[string]any, 0, len(definition.Nodes))
	for _, node := range definition.Nodes {
		entry := map[string]any{"id": node.ID, "module": node.Module, "state": "pending"}
		if receipt, recorded := states[node.ID]; recorded {
			entry["state"] = string(receipt.State)
			entry["attempt"] = receipt.Attempt
			entry["receipt"] = idText(receipt.ID)
			if receipt.Failure != "" {
				entry["failure"] = receipt.Failure
			}
		}
		nodes = append(nodes, entry)
	}
	edges := make([]map[string]any, 0, len(definition.Edges))
	for _, edge := range definition.Edges {
		edges = append(edges, map[string]any{"from": edge.From.Node, "to": edge.To.Node})
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"operation": id, "recipe": recipeID, "nodes": nodes, "edges": edges,
	})
}

// operationInbox projects every operation waiting on an operator: the
// blocked state, the advertised recovery actions grant/decline resolve,
// and any prior committed decision on the same operation chain. One
// list answers "what is waiting on me" instead of the operator polling
// individual operations.
func (h *Handler) operationInbox(response http.ResponseWriter, request *http.Request) {
	items := []map[string]any{}
	for _, status := range h.operations.List() {
		if status.State != operation.StateBlocked || status.Recovery == nil {
			continue
		}
		item := map[string]any{
			"operation": status.ID, "task": status.Task, "recipe": status.Recipe,
			"reason": status.Recovery.Reason, "subject": status.Recovery.Subject,
			"actions": status.Recovery.Actions, "evidence": status.Recovery.Evidence,
		}
		if h.repository != nil {
			if prior, found, err := runrecord.ResolveHumanDecision(request.Context(), h.repository, status.ID); err == nil && found {
				item["prior_decision"] = map[string]any{
					"id": prior.ID, "answer": prior.Answer, "tool": prior.Tool,
				}
			}
		}
		items = append(items, item)
	}
	writeJSON(response, http.StatusOK, map[string]any{"waiting": items})
}

func (h *Handler) operationCancel(response http.ResponseWriter, request *http.Request) {
	var body operationCancelRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if !h.operations.Cancel(body.ID) {
		writeError(response, http.StatusConflict, "operation_not_active", "operation is unknown or terminal")
		return
	}
	status, _ := h.operations.Status(body.ID)
	writeJSON(response, http.StatusAccepted, status)
}

func (h *Handler) operationWait(response http.ResponseWriter, request *http.Request) {
	id, err := artifact.ParseID(request.URL.Query().Get("id"))
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	status, err := h.operations.Wait(request.Context(), id)
	if err != nil {
		if request.Context().Err() != nil {
			writeGenerationError(response, err)
		} else {
			writeError(response, http.StatusNotFound, "not_found", "operation not found")
		}
		return
	}
	writeJSON(response, http.StatusOK, status)
}

func (h *Handler) operationDecision(response http.ResponseWriter, request *http.Request) {
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "decision repository is unavailable")
		return
	}
	var body operationDecisionRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	status, found := h.operations.Status(body.Operation)
	if !found || status.State != operation.StateBlocked || status.Recovery == nil {
		writeError(response, http.StatusConflict, "operation_not_blocked", "operation has no blocked action")
		return
	}
	var action operatoraction.Action
	for _, candidate := range status.Recovery.Actions {
		if candidate.Code == body.Tool {
			action = candidate
			break
		}
	}
	if action.Code == "" || body.Answer != operatoraction.AnswerGrant && body.Answer != operatoraction.AnswerDecline {
		writeInvalidRequest(response, errors.New("decision requires an advertised tool and valid answer"))
		return
	}
	prior, found, err := runrecord.ResolveHumanDecision(request.Context(), h.repository, body.Operation)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	priorID := artifact.ID{}
	if found {
		priorID = prior.ID
	}
	approval, err := operatoraction.NewApprovalRequest(body.Operation, status.Recipe, action, priorID)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	decision, err := runrecord.NewHumanDecision(approval, body.Answer)
	if err == nil {
		err = runrecord.PublishHumanDecision(request.Context(), h.repository, approval, decision)
	}
	if err != nil {
		writeError(response, http.StatusConflict, "decision_conflict", err.Error())
		return
	}
	operationID := body.Operation
	if body.Answer == operatoraction.AnswerGrant {
		operationID, err = h.operations.RecoverAfterDecision(context.WithoutCancel(request.Context()), decision)
		if err != nil {
			writeError(response, http.StatusConflict, "decision_recovery_failed", err.Error())
			return
		}
	}
	writeJSON(response, http.StatusAccepted, operationDecisionResponse{Decision: decision.ID, Operation: operationID})
}
