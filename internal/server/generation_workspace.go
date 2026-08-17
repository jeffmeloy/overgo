package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

type GenerationControlType string

const (
	GenerationControlText    GenerationControlType = "text"
	GenerationControlInteger GenerationControlType = "integer"
	GenerationControlNumber  GenerationControlType = "number"
	GenerationControlBoolean GenerationControlType = "boolean"
)

type GenerationControl struct {
	Name     string                `json:"name"`
	Type     GenerationControlType `json:"type"`
	Required bool                  `json:"required,omitempty"`
}

type GenerationCapability struct {
	Task     recipe.Task         `json:"task"`
	Recipe   artifact.ID         `json:"recipe"`
	Stages   []recipe.Stage      `json:"stages"`
	Inputs   []recipe.Input      `json:"inputs,omitempty"`
	Outputs  []recipe.Output     `json:"outputs"`
	Controls []GenerationControl `json:"controls"`
}

type GenerationWorkspaceAPI interface {
	GenerationCapabilities(context.Context) ([]GenerationCapability, error)
	ExecuteGeneration(context.Context, recipe.Task, artifact.ID, json.RawMessage, operation.Reporter) (operation.Completion, error)
}

type generationRequest struct {
	Task   recipe.Task     `json:"task"`
	Recipe artifact.ID     `json:"recipe"`
	Input  json.RawMessage `json:"input"`
}

type generationResponse struct {
	Operation artifact.ID `json:"operation"`
}

func (h *Handler) generationCapabilities(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	workspace, ok := h.generator.(GenerationWorkspaceAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "generation workspace is unavailable")
		return
	}
	capabilities, err := workspace.GenerationCapabilities(request.Context())
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	if err := validateGenerationCapabilities(capabilities); err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, capabilities)
}

func (h *Handler) generationRun(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	workspace, ok := h.generator.(GenerationWorkspaceAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "generation workspace is unavailable")
		return
	}
	var body generationRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	capabilities, err := workspace.GenerationCapabilities(request.Context())
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	capability, err := selectGenerationCapability(capabilities, body.Task, body.Recipe)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	if err := validateGenerationInput(capability.Controls, body.Input); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	id, err := h.operations.Submit(context.WithoutCancel(request.Context()), operation.Request{
		Task: body.Task, Recipe: body.Recipe,
	}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return workspace.ExecuteGeneration(ctx, body.Task, body.Recipe, body.Input, reporter)
	})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusAccepted, generationResponse{Operation: id})
}

func validateGenerationCapabilities(capabilities []GenerationCapability) error {
	seen := make(map[recipe.Task]bool, len(capabilities))
	for _, capability := range capabilities {
		if !capability.Task.Valid() || capability.Recipe.Kind() != artifact.KindRecipe || len(capability.Stages) == 0 {
			return errors.New("generation workspace: invalid runtime capability")
		}
		if seen[capability.Task] {
			return fmt.Errorf("generation workspace: duplicate task %q", capability.Task)
		}
		seen[capability.Task] = true
		fields := make(map[string]bool, len(capability.Controls))
		for _, control := range capability.Controls {
			if control.Name == "" || !control.Type.valid() || fields[control.Name] {
				return fmt.Errorf("generation workspace: invalid control %q", control.Name)
			}
			fields[control.Name] = true
		}
	}
	return nil
}

func (kind GenerationControlType) valid() bool {
	return kind == GenerationControlText || kind == GenerationControlInteger ||
		kind == GenerationControlNumber || kind == GenerationControlBoolean
}

func selectGenerationCapability(capabilities []GenerationCapability, task recipe.Task, recipeID artifact.ID) (GenerationCapability, error) {
	if err := validateGenerationCapabilities(capabilities); err != nil {
		return GenerationCapability{}, err
	}
	for _, capability := range capabilities {
		if capability.Task == task && capability.Recipe == recipeID {
			return capability, nil
		}
	}
	return GenerationCapability{}, errors.New("generation workspace: task and recipe are not admitted")
}

func validateGenerationInput(controls []GenerationControl, raw json.RawMessage) error {
	fields := make(map[string]json.RawMessage, len(controls))
	if err := strictjson.Decode(bytes.NewReader(raw), &fields); err != nil {
		return fmt.Errorf("generation workspace: invalid input: %w", err)
	}
	declarations := make(map[string]GenerationControl, len(controls))
	for _, control := range controls {
		declarations[control.Name] = control
		if control.Required {
			if _, ok := fields[control.Name]; !ok {
				return fmt.Errorf("generation workspace: input %q is required", control.Name)
			}
		}
	}
	for name, rawValue := range fields {
		control, ok := declarations[name]
		if !ok {
			return fmt.Errorf("generation workspace: input %q is undeclared", name)
		}
		if err := validateGenerationValue(control.Type, rawValue); err != nil {
			return fmt.Errorf("generation workspace: input %q: %w", name, err)
		}
	}
	return nil
}

func validateGenerationValue(kind GenerationControlType, raw json.RawMessage) error {
	var target any
	switch kind {
	case GenerationControlText:
		target = new(string)
	case GenerationControlInteger:
		target = new(int64)
	case GenerationControlNumber:
		target = new(float64)
	case GenerationControlBoolean:
		target = new(bool)
	default:
		return errors.New("invalid control type")
	}
	if err := strictjson.Decode(bytes.NewReader(raw), target); err != nil {
		return err
	}
	if number, ok := target.(*float64); ok && (math.IsNaN(*number) || math.IsInf(*number, 0)) {
		return errors.New("number must be finite")
	}
	return nil
}
