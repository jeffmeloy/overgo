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

// The automation routes: each serves one path of the route table over the automation workspace (automationRoute).

func (h *Handler) automationInventory(workspace AutomationWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	inventory, err := workspace.AutomationInventory(request.Context())
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, inventory)
}

func (h *Handler) automationDefine(workspace AutomationWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body AutomationDefinitionInput
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	definition, err := workspace.PublishAutomationDefinition(request.Context(), body)
	writeAutomationResult(response, http.StatusCreated, struct {
		ID         artifact.ID                 `json:"id"`
		Definition recipe.AutomationDefinition `json:"definition"`
	}{ID: definition.ID, Definition: definition}, err)
}

func (h *Handler) automationActivate(workspace AutomationWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body automationActivationRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	active, err := workspace.ActivateAutomation(request.Context(), body.Definition)
	writeAutomationResult(response, http.StatusOK, active, err)
}

func (h *Handler) automationRun(workspace AutomationWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body AutomationExecutionInput
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	execution, err := workspace.RunAutomation(context.WithoutCancel(request.Context()), h.operations, body)
	writeAutomationResult(response, http.StatusAccepted, execution, err)
}

func (h *Handler) automationSchedule(workspace AutomationWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body AutomationExecutionInput
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	execution, fired, err := workspace.ScheduleAutomation(context.WithoutCancel(request.Context()), h.operations, body)
	writeAutomationResult(response, http.StatusAccepted, struct {
		Execution workflowruntime.AutomationExecution `json:"execution"`
		Fired     bool                                `json:"fired"`
	}{Execution: execution, Fired: fired}, err)
}

func (h *Handler) automationHistory(workspace AutomationWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	history, err := workspace.AutomationHistory(request.Context())
	writeAutomationResult(response, http.StatusOK, history, err)
}

func writeAutomationResult(response http.ResponseWriter, status int, value any, err error) {
	writeRefusableResult(response, status, value, err, "automation_refused")
}
