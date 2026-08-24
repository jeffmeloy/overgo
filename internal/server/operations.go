package server

import (
	"context"
	"errors"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
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
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
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

func (h *Handler) operationCancel(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
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
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
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
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
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
