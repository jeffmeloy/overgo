package server

import (
	"context"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

type automationActivationRequest struct {
	Definition artifact.ID `json:"definition"`
}

func (h *Handler) automationWorkspace(response http.ResponseWriter, request *http.Request) {
	workspace, ok := h.generator.(AutomationWorkspaceAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "automation workspace is unavailable")
		return
	}
	switch request.URL.Path {
	case "/automations":
		if !requireMethod(response, request, http.MethodGet) {
			return
		}
		inventory, err := workspace.AutomationInventory(request.Context())
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		writeJSON(response, http.StatusOK, inventory)
	case "/automations/definitions":
		var body AutomationDefinitionInput
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		definition, err := workspace.PublishAutomationDefinition(request.Context(), body)
		writeAutomationResult(response, http.StatusCreated, struct {
			ID         artifact.ID                 `json:"id"`
			Definition recipe.AutomationDefinition `json:"definition"`
		}{ID: definition.ID, Definition: definition}, err)
	case "/automations/activate":
		var body automationActivationRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		active, err := workspace.ActivateAutomation(request.Context(), body.Definition)
		writeAutomationResult(response, http.StatusOK, active, err)
	case "/automations/run":
		var body AutomationExecutionInput
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		execution, err := workspace.RunAutomation(context.WithoutCancel(request.Context()), h.operations, body)
		writeAutomationResult(response, http.StatusAccepted, execution, err)
	case "/automations/schedule":
		var body AutomationExecutionInput
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		execution, fired, err := workspace.ScheduleAutomation(context.WithoutCancel(request.Context()), h.operations, body)
		writeAutomationResult(response, http.StatusAccepted, struct {
			Execution workflowruntime.AutomationExecution `json:"execution"`
			Fired     bool                                `json:"fired"`
		}{Execution: execution, Fired: fired}, err)
	case "/automations/history":
		if !requireMethod(response, request, http.MethodGet) {
			return
		}
		history, err := workspace.AutomationHistory(request.Context())
		writeAutomationResult(response, http.StatusOK, history, err)
	default:
		writeError(response, http.StatusNotFound, "not_found", "automation route is absent")
	}
}

func writeAutomationResult(response http.ResponseWriter, status int, value any, err error) {
	writeRefusableResult(response, status, value, err, "automation_refused")
}
