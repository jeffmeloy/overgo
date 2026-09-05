package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

type WorkflowControlType string

const (
	WorkflowControlText       WorkflowControlType = "text"
	WorkflowControlInteger    WorkflowControlType = "integer"
	WorkflowControlNumber     WorkflowControlType = "number"
	WorkflowControlBoolean    WorkflowControlType = "boolean"
	WorkflowControlDataset    WorkflowControlType = "dataset"
	WorkflowControlCheckpoint WorkflowControlType = "checkpoint"
	WorkflowControlOutput     WorkflowControlType = "managed-output"
)

type WorkflowControl struct {
	Name     string              `json:"name"`
	Type     WorkflowControlType `json:"type"`
	Required bool                `json:"required,omitzero"`
}

type WorkflowCapability struct {
	Task     recipe.Task       `json:"task"`
	Recipe   artifact.ID       `json:"recipe"`
	Stages   []recipe.Stage    `json:"stages"`
	Inputs   []recipe.Input    `json:"inputs,omitempty"`
	Outputs  []recipe.Output   `json:"outputs"`
	Controls []WorkflowControl `json:"controls"`
	// Model and Name identify the activated model behind a store-derived
	// capability, so a task served by several models lists each of them;
	// Refusal names why the page cannot run this one (its request carries
	// what no page can type), and such a capability is listed, not run.
	Model    artifact.ID `json:"model,omitzero"`
	Name     string      `json:"name,omitzero"`
	Location string      `json:"location,omitzero"`
	Refusal  string      `json:"refusal,omitzero"`
}

type WorkflowKind string

const (
	WorkflowGeneration WorkflowKind = "generation"
	WorkflowTraining   WorkflowKind = "training"
	WorkflowExport     WorkflowKind = "export"
	WorkflowModelBuild WorkflowKind = "model-builder"
)

type WorkflowWorkspaceAPI interface {
	WorkflowCapabilities(context.Context, WorkflowKind) ([]WorkflowCapability, error)
	ExecuteWorkflow(context.Context, WorkflowKind, recipe.Task, artifact.ID, json.RawMessage, operation.Reporter) (operation.Completion, error)
}

type WorkflowWorkspaceSet []WorkflowWorkspaceAPI

func (set WorkflowWorkspaceSet) WorkflowCapabilities(ctx context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	var capabilities []WorkflowCapability
	for _, workspace := range set {
		current, err := workspace.WorkflowCapabilities(ctx, kind)
		if err != nil {
			return nil, err
		}
		capabilities = append(capabilities, current...)
	}
	return capabilities, nil
}

func (set WorkflowWorkspaceSet) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	for _, workspace := range set {
		capabilities, err := workspace.WorkflowCapabilities(ctx, kind)
		if err != nil {
			return operation.Completion{}, err
		}
		if _, err = selectWorkflowCapability(capabilities, task, recipeID); err == nil {
			return workspace.ExecuteWorkflow(ctx, kind, task, recipeID, raw, reporter)
		}
	}
	return operation.Completion{}, errors.New("workflow workspace: task and recipe are not admitted")
}

type workflowRequest struct {
	Task   recipe.Task     `json:"task"`
	Recipe artifact.ID     `json:"recipe"`
	Input  json.RawMessage `json:"input"`
}

type workflowResponse struct {
	Operation artifact.ID `json:"operation"`
}

func (h *Handler) workflowCapabilities(response http.ResponseWriter, request *http.Request, kind WorkflowKind) {
	workspace, ok := h.generator.(WorkflowWorkspaceAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, string(kind)+" workspace is unavailable")
		return
	}
	capabilities, err := workspace.WorkflowCapabilities(request.Context(), kind)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	if len(capabilities) == 0 {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, string(kind)+" workspace is unavailable")
		return
	}
	if err := validateWorkflowCapabilities(capabilities); err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, capabilities)
}

func (h *Handler) workflowRun(response http.ResponseWriter, request *http.Request, kind WorkflowKind) {
	workspace, ok := h.generator.(WorkflowWorkspaceAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, string(kind)+" workspace is unavailable")
		return
	}
	var body workflowRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	capabilities, err := workspace.WorkflowCapabilities(request.Context(), kind)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	capability, err := selectWorkflowCapability(capabilities, body.Task, body.Recipe)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	if err := validateWorkflowInput(capability.Controls, body.Input); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	id, err := h.submitWorkflow(context.WithoutCancel(request.Context()), workspace, kind, capability, body.Input)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusAccepted, workflowResponse{Operation: id})
}

func (h *Handler) submitWorkflow(
	ctx context.Context,
	workspace WorkflowWorkspaceAPI,
	kind WorkflowKind,
	capability WorkflowCapability,
	input json.RawMessage,
) (artifact.ID, error) {
	if h.repository == nil {
		return artifact.ID{}, errors.New("workflow workspace: durable repository required")
	}
	request := operation.Request{Task: capability.Task, Recipe: capability.Recipe}
	intent, err := artifact.JSONID(artifact.KindEvidence, struct {
		Version uint16          `json:"version"`
		Kind    WorkflowKind    `json:"kind"`
		Task    recipe.Task     `json:"task"`
		Recipe  artifact.ID     `json:"recipe"`
		Input   json.RawMessage `json:"input"`
	}{artifact.InitialDocumentVersion, kind, capability.Task, capability.Recipe, input})
	if err != nil {
		return artifact.ID{}, err
	}
	execute := func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return h.executeObservedOperation(ctx, reporter, capability.Task, capability.Recipe,
			func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
				return operation.ExecuteReentrant(ctx, h.repository, reporter, request, intent,
					func(ctx context.Context) (operation.Completion, error) {
						return workspace.ExecuteWorkflow(ctx, kind, capability.Task, capability.Recipe, input, reporter)
					})
			})
	}
	return h.operations.Submit(ctx, request, execute)
}

func validateWorkflowCapabilities(capabilities []WorkflowCapability) error {
	// A task is served by every model activated for it, so the recipe is
	// the identity; a task listed twice under one recipe is the fault.
	seen := make(map[artifact.ID]bool, len(capabilities))
	for _, capability := range capabilities {
		if !capability.Task.Valid() || capability.Recipe.Kind() != artifact.KindRecipe || len(capability.Stages) == 0 {
			return errors.New("workflow workspace: invalid runtime capability")
		}
		if seen[capability.Recipe] {
			return fmt.Errorf("workflow workspace: duplicate capability %s for %q", capability.Recipe, capability.Task)
		}
		seen[capability.Recipe] = true
		fields := make(map[string]bool, len(capability.Controls))
		for _, control := range capability.Controls {
			if control.Name == "" || !control.Type.valid() || fields[control.Name] {
				return fmt.Errorf("workflow workspace: invalid control %q", control.Name)
			}
			fields[control.Name] = true
		}
	}
	return nil
}

func (kind WorkflowControlType) valid() bool {
	return kind == WorkflowControlText || kind == WorkflowControlInteger ||
		kind == WorkflowControlNumber || kind == WorkflowControlBoolean ||
		kind == WorkflowControlDataset || kind == WorkflowControlCheckpoint ||
		kind == WorkflowControlOutput
}

func selectWorkflowCapability(capabilities []WorkflowCapability, task recipe.Task, recipeID artifact.ID) (WorkflowCapability, error) {
	if err := validateWorkflowCapabilities(capabilities); err != nil {
		return WorkflowCapability{}, err
	}
	for _, capability := range capabilities {
		if capability.Task == task && capability.Recipe == recipeID {
			return capability, nil
		}
	}
	return WorkflowCapability{}, errors.New("workflow workspace: task and recipe are not admitted")
}

func validateWorkflowInput(controls []WorkflowControl, raw json.RawMessage) error {
	fields := make(map[string]json.RawMessage, len(controls))
	if err := strictjson.Decode(bytes.NewReader(raw), &fields); err != nil {
		return fmt.Errorf("workflow workspace: invalid input: %w", err)
	}
	declarations := make(map[string]WorkflowControl, len(controls))
	for _, control := range controls {
		declarations[control.Name] = control
		if control.Required {
			if _, ok := fields[control.Name]; !ok {
				return fmt.Errorf("workflow workspace: input %q is required", control.Name)
			}
		}
	}
	for name, rawValue := range fields {
		control, ok := declarations[name]
		if !ok {
			return fmt.Errorf("workflow workspace: input %q is undeclared", name)
		}
		if err := validateWorkflowValue(control.Type, rawValue); err != nil {
			return fmt.Errorf("workflow workspace: input %q: %w", name, err)
		}
	}
	return nil
}

func validateWorkflowValue(controlType WorkflowControlType, raw json.RawMessage) error {
	var target any
	switch controlType {
	case WorkflowControlText, WorkflowControlOutput:
		target = new(string)
	case WorkflowControlDataset, WorkflowControlCheckpoint:
		var value string
		if err := strictjson.Decode(bytes.NewReader(raw), &value); err != nil {
			return err
		}
		id, err := artifact.ParseID(value)
		artifactKind := artifact.KindDataset
		if controlType == WorkflowControlCheckpoint {
			artifactKind = artifact.KindCheckpoint
		}
		if err != nil || id.Kind() != artifactKind {
			return fmt.Errorf("%s artifact required", artifactKind)
		}
		return nil
	case WorkflowControlInteger:
		target = new(int64)
	case WorkflowControlNumber:
		target = new(float64)
	case WorkflowControlBoolean:
		target = new(bool)
	default:
		return errors.New("invalid control type")
	}
	if err := strictjson.Decode(bytes.NewReader(raw), target); err != nil {
		return err
	}
	if number, ok := target.(*float64); ok && !checked.Finite64(*number) {
		return errors.New("number must be finite")
	}
	return nil
}
