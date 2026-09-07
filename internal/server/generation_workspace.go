package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
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
	// WorkflowControlArtifact names an artifact the store holds; a page
	// fills it from a media card's stored id rather than typing it.
	WorkflowControlArtifact WorkflowControlType = "artifact"
)

type WorkflowControl struct {
	Name     string              `json:"name"`
	Type     WorkflowControlType `json:"type"`
	Required bool                `json:"required,omitzero"`
	// Choices are the values the model's artifact exports for the field;
	// a page offers them instead of a free input.
	Choices []string `json:"choices,omitempty"`
	// Label and Media describe an artifact slot: what a page calls it and
	// the kind of stored file it takes (image, video, audio), so the page
	// lists matching intakes beside it and routes an attachment to it.
	Label string `json:"label,omitzero"`
	Media string `json:"media,omitzero"`
	// Bounds are a numeric control's declared default, step and rate, from
	// which a page derives its presets (aspect ratios, durations).
	Bounds *WorkflowControlBounds `json:"bounds,omitempty"`
}

// WorkflowControlBounds is a numeric control's declared default, the step a
// valid value moves by (zero when any value is valid) and, for a frame
// count, the frames one second holds.
type WorkflowControlBounds struct {
	Default int `json:"default"`
	Step    int `json:"step,omitzero"`
	Rate    int `json:"rate,omitzero"`
}

// slotMedia: the media kinds an artifact slot may declare.
var slotMedia = map[string]bool{"": true, "image": true, "video": true, "audio": true}

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
	// model is projected from the compiled recipe by native workspace owners.
	// It must not be inferred from an unrelated co-hosted text runner.
	model artifact.ID
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

func failWorkflow(ctx context.Context, store artifact.Repository, recipeID artifact.ID, inputs []artifact.ID, failure string, cause error) (operation.Completion, error) {
	outcome := runrecord.OutcomeFailed
	if errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) {
		outcome, failure = runrecord.OutcomeCancelled, ""
	}
	run, err := runrecord.NewRun(recipeID, outcome, inputs, nil, failure)
	if err == nil {
		batch, batchErr := run.Batch("workflow/failed-run/" + run.ID.String())
		if batchErr == nil {
			_, batchErr = artifact.CommitBatch(context.WithoutCancel(ctx), store, batch)
			if errors.Is(batchErr, artifact.ErrNoChange) {
				batchErr = nil
			}
		}
		err = batchErr
	}
	return operation.Completion{Run: run.ID}, errors.Join(cause, err)
}

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
	// Sources are stored documents the run cites as inputs beside its
	// request: a prompt enhancement the page accepted keeps the original
	// prompt as the request's source. Each must be an artifact the
	// repository holds.
	Sources []artifact.ID `json:"sources,omitempty"`
}

type workflowResponse struct {
	Operation artifact.ID `json:"operation"`
}

// workflowSourcesKey carries a submission's sources to the workspace that
// records the run, since the workflow API takes the request bytes alone.
type workflowSourcesKey struct{}

// withWorkflowSources binds the sources a run cites to the context its
// execution receives.
func withWorkflowSources(ctx context.Context, sources []artifact.ID) context.Context {
	if len(sources) == 0 {
		return ctx
	}
	return context.WithValue(ctx, workflowSourcesKey{}, slices.Clone(sources))
}

// workflowSources reads the sources bound to an execution's context.
func workflowSources(ctx context.Context) []artifact.ID {
	sources, _ := ctx.Value(workflowSourcesKey{}).([]artifact.ID)
	return sources
}

// validateWorkflowSources requires every source to be an artifact the
// repository holds.
func validateWorkflowSources(ctx context.Context, repository *overgodb.Store, sources []artifact.ID) error {
	if len(sources) > 0 && repository == nil {
		return errors.New("workflow workspace: sources need a durable repository")
	}
	for _, source := range sources {
		_, found, err := repository.Artifact(ctx, source)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("workflow workspace: source %s is not stored", source)
		}
	}
	return nil
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
	if err := validateWorkflowSources(request.Context(), h.repository, body.Sources); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	id, err := h.submitWorkflow(context.WithoutCancel(request.Context()), workspace, kind, capability, body.Input, body.Sources)
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
	sources []artifact.ID,
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
		Sources []artifact.ID   `json:"sources,omitempty"`
	}{artifact.InitialDocumentVersion, kind, capability.Task, capability.Recipe, input, sources})
	if err != nil {
		return artifact.ID{}, err
	}
	execute := func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return h.executeObservedOperation(ctx, reporter, capability.Task, capability.Recipe, capability.model,
			func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
				return operation.ExecuteReentrant(ctx, h.repository, reporter, request, intent,
					func(ctx context.Context) (operation.Completion, error) {
						return workspace.ExecuteWorkflow(withWorkflowSources(ctx, sources), kind, capability.Task, capability.Recipe, input, reporter)
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
			if control.Name == "" || !control.Type.valid() || fields[control.Name] || !slotMedia[control.Media] ||
				control.Bounds != nil && (control.Bounds.Step < 0 || control.Bounds.Rate < 0) {
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
		kind == WorkflowControlOutput || kind == WorkflowControlArtifact
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
	case WorkflowControlArtifact:
		// The page fills an artifact control from a media card's stored id;
		// a value that is no artifact id is refused before any run.
		var value string
		if err := strictjson.Decode(bytes.NewReader(raw), &value); err != nil {
			return err
		}
		_, err := artifact.ParseID(value)
		return err
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
