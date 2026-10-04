package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
)

type workflowControlType string

const (
	workflowControlText       workflowControlType = "text"
	workflowControlInteger    workflowControlType = "integer"
	workflowControlNumber     workflowControlType = "number"
	workflowControlBoolean    workflowControlType = "boolean"
	workflowControlDataset    workflowControlType = "dataset"
	workflowControlCheckpoint workflowControlType = "checkpoint"
	workflowControlOutput     workflowControlType = "managed-output"
	// workflowControlArtifact names an artifact the store holds; a page
	// fills it from a media card's stored id rather than typing it.
	workflowControlArtifact workflowControlType = "artifact"
)

type workflowControl struct {
	Name     string              `json:"name"`
	Type     workflowControlType `json:"type"`
	Required bool                `json:"required,omitzero"`
	// Choices are the values the model's artifact exports for the field;
	// a page offers them instead of a free input.
	Choices []string `json:"choices,omitempty"`
	// Label and Media describe an artifact slot: what a page calls it and
	// the kind of stored file it takes (image, video, audio), so the page
	// lists matching intakes beside it and routes an attachment to it.
	Label string `json:"label,omitzero"`
	Media string `json:"media,omitzero"`
	// Format is the catalog format a dataset control takes: a page offers the
	// catalogued datasets that hold it by name. A dataset control with none
	// takes an identity typed in, as a dataset the catalog does not hold needs.
	Format string `json:"format,omitzero"`
	// Bounds are a numeric control's declared default, step and rate, from
	// which a page derives its presets (aspect ratios, durations).
	Bounds *workflowControlBounds               `json:"bounds,omitempty"`
	Audio  *speechrecognition.InputRequirements `json:"audio,omitempty"`
}

func audioFormatMessage(audio *speechrecognition.InputRequirements) string {
	if audio == nil {
		return "Audio format does not match this model. Attach a compatible audio file."
	}
	channels := fmt.Sprintf("%d-channel", audio.Format.Channels)
	if audio.Format.Channels == 1 {
		channels = "mono"
	} else if audio.Format.Channels == 2 {
		channels = "stereo"
	}
	return fmt.Sprintf("This model needs %d Hz %s audio. Record in Transcribe or attach a compatible audio file.", audio.Format.SampleRate, channels)
}

// workflowControlBounds is a numeric control's declared default, the step a
// valid value moves by (zero when any value is valid) and, for a frame
// count, the frames one second holds.
type workflowControlBounds struct {
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
	Controls []workflowControl `json:"controls"`
	// Model and Name identify the activated model behind a store-derived
	// capability, so a task served by several models lists each of them;
	// Refusal names why the page cannot run this one (its request carries
	// what no page can type), and such a capability is listed, not run.
	Model    artifact.ID `json:"model,omitzero"`
	Name     string      `json:"name,omitzero"`
	Location string      `json:"location,omitzero"`
	Refusal  string      `json:"refusal,omitzero"`
	// Host: the run executes on the host processor and needs no device memory.
	Host bool `json:"host,omitzero"`
	// Note states what the page cannot derive from the controls, such as how
	// a run learns from its dataset.
	Note string `json:"note,omitzero"`
	// model is projected from the compiled recipe by native workspace owners.
	// It must not be inferred from an unrelated co-hosted text runner.
	model artifact.ID
	// workspace is the member of a workspace set that listed the capability,
	// so its run goes there without listing again.
	workspace WorkflowWorkspaceAPI
	// execution and directory are what a store generation listing resolved,
	// so its run resolves neither again.
	execution modelrecipe.CapabilityEvidenceSelection
	directory string
	// Native streaming remains attached to the selected workspace capability;
	// aggregating workspaces does not create a separate audio registry.
	transcriptionStream func(context.Context, speechrecognition.StreamRequest) (*transcriptionHTTPStream, error)
}

type WorkflowKind string

const (
	workflowGeneration WorkflowKind = "generation"
	WorkflowTraining   WorkflowKind = "training"
	workflowExport     WorkflowKind = "export"
	workflowModelBuild WorkflowKind = "model-builder"
)

type WorkflowWorkspaceAPI interface {
	WorkflowCapabilities(context.Context, WorkflowKind) ([]WorkflowCapability, error)
	// ExecuteWorkflow runs a capability the workspace listed, as listed.
	ExecuteWorkflow(context.Context, WorkflowKind, WorkflowCapability, json.RawMessage, operation.Reporter) (operation.Completion, error)
}

// workflowPreviewAPI is a workflow workspace that shows one page of a run's
// input as the run would read it, before any run.
type workflowPreviewAPI interface {
	PreviewWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, position, limit int) (any, error)
}

type WorkflowWorkspaceSet []WorkflowWorkspaceAPI

// PreviewWorkflow asks the workspace that admits the task and recipe.
func (set WorkflowWorkspaceSet) PreviewWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, position, limit int) (any, error) {
	for _, workspace := range set {
		capabilities, err := workspace.WorkflowCapabilities(ctx, kind)
		if err != nil {
			return nil, err
		}
		if _, err = selectWorkflowCapability(capabilities, task, recipeID); err != nil {
			continue
		}
		return previewWorkflowIn(ctx, workspace, kind, task, recipeID, raw, position, limit)
	}
	return nil, errors.New("workflow workspace: task and recipe are not admitted")
}

// errWorkflowPreviewUnavailable: the workflow reads its input only when it runs.
var errWorkflowPreviewUnavailable = errors.New("workflow workspace: this workflow has no input preview")

// previewWorkflowIn asks a workspace for a preview.
func previewWorkflowIn(ctx context.Context, workspace WorkflowWorkspaceAPI, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, position, limit int) (any, error) {
	previewer, ok := workspace.(workflowPreviewAPI)
	if !ok {
		return nil, errWorkflowPreviewUnavailable
	}
	return previewer.PreviewWorkflow(ctx, kind, task, recipeID, raw, position, limit)
}

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
		for _, capability := range current {
			capability.workspace = cmp.Or(capability.workspace, workspace)
			capabilities = append(capabilities, capability)
		}
	}
	return capabilities, nil
}

// ExecuteWorkflow runs the capability in the member workspace that listed it.
func (set WorkflowWorkspaceSet) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, capability WorkflowCapability, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if capability.workspace == nil {
		return operation.Completion{}, errors.New("workflow workspace: task and recipe are not admitted")
	}
	return capability.workspace.ExecuteWorkflow(ctx, kind, capability, raw, reporter)
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
	if len(sources) > 0 {
		if err := repository.Refresh(ctx); err != nil {
			return err
		}
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

// workflowRoute serves one workflow kind's route over the workflow
// workspace; a launch without one answers 501 with what is missing.
func workflowRoute(kind WorkflowKind, serve func(*Handler, http.ResponseWriter, *http.Request, WorkflowKind)) routeHandler {
	return func(h *Handler, response http.ResponseWriter, request *http.Request) {
		if h.config.Workflows == nil {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, string(kind)+" workspace is unavailable")
			return
		}
		serve(h, response, request, kind)
	}
}

func (h *Handler) workflowCapabilities(response http.ResponseWriter, request *http.Request, kind WorkflowKind) {
	capabilities, err := h.config.Workflows.WorkflowCapabilities(request.Context(), kind)
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

// admitWorkflow decodes a submission and holds it to the checks a run
// makes: an active capability, its controls, stored sources; a refusal
// is written.
func (h *Handler) admitWorkflow(response http.ResponseWriter, request *http.Request, kind WorkflowKind) (workflowRequest, WorkflowCapability, bool) {
	var body workflowRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return body, WorkflowCapability{}, false
	}
	capabilities, err := h.config.Workflows.WorkflowCapabilities(request.Context(), kind)
	if err != nil {
		writeGenerationError(response, err)
		return body, WorkflowCapability{}, false
	}
	capability, err := selectWorkflowCapability(capabilities, body.Task, body.Recipe)
	if err != nil {
		writeInvalidRequest(response, err)
		return body, WorkflowCapability{}, false
	}
	if err := validateWorkflowInput(capability.Controls, body.Input); err != nil {
		writeInvalidRequest(response, err)
		return body, WorkflowCapability{}, false
	}
	if err := validateWorkflowSources(request.Context(), h.repository, body.Sources); err != nil {
		writeInvalidRequest(response, err)
		return body, WorkflowCapability{}, false
	}
	return body, capability, true
}

func (h *Handler) workflowRun(response http.ResponseWriter, request *http.Request, kind WorkflowKind) {
	body, capability, admitted := h.admitWorkflow(response, request, kind)
	if !admitted {
		return
	}
	id, err := h.submitWorkflow(context.WithoutCancel(request.Context()), kind, capability, body.Input, body.Sources)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusAccepted, workflowResponse{Operation: id})
}

// workflowCommand is one way to reproduce a submission outside the page:
// "cli" runs the recipe from a command, "http" posts the submission again.
type workflowCommand struct {
	Kind string `json:"kind"`
	Text string `json:"text"`
}

// workflowCommands answers the commands that reproduce an admitted
// submission without running it: a generation recipe also runs from
// cmd/recipe on the same executor catalog and store, and every submission
// runs again by posting it to this server.
func (h *Handler) workflowCommands(response http.ResponseWriter, request *http.Request, kind WorkflowKind) {
	body, capability, admitted := h.admitWorkflow(response, request, kind)
	if !admitted {
		return
	}
	submission, err := json.Marshal(body)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	commands := []workflowCommand{}
	if kind == workflowGeneration && capability.Location != "" {
		commands = append(commands, workflowCommand{Kind: "cli", Text: "go run ./cmd/recipe run -task " + shellQuote(string(body.Task)) +
			" -input " + shellQuote(string(body.Input)) + " " + shellQuote(capability.Location)})
	}
	scheme, authorization := "http", ""
	if request.TLS != nil {
		scheme = "https"
	}
	if h.config.APIKey != "" {
		authorization = ` -H "Authorization: Bearer $OVERGO_API_KEY"`
	}
	commands = append(commands, workflowCommand{Kind: "http", Text: "curl -X POST " + shellQuote(scheme+"://"+request.Host+"/"+string(kind)+"/run") +
		" -H 'Content-Type: application/json'" + authorization + " -d " + shellQuote(string(submission))})
	writeJSON(response, http.StatusOK, map[string]any{"commands": commands})
}

// shellQuote quotes a word for a POSIX shell.
func shellQuote(word string) string {
	return "'" + strings.ReplaceAll(word, "'", `'\''`) + "'"
}

type workflowPreviewRequest struct {
	Task     recipe.Task     `json:"task"`
	Recipe   artifact.ID     `json:"recipe"`
	Input    json.RawMessage `json:"input"`
	Position int             `json:"position"`
	Limit    int             `json:"limit"`
}

// workflowPreview answers one page of a run's input as the workflow reads it;
// the input need hold only what the preview reads.
func (h *Handler) workflowPreview(response http.ResponseWriter, request *http.Request, kind WorkflowKind) {
	var body workflowPreviewRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if body.Position < 0 || body.Limit <= 0 || body.Limit > h.config.MaxStoredResponses {
		writeInvalidRequestMessage(response, "invalid workflow preview page")
		return
	}
	preview, err := previewWorkflowIn(request.Context(), h.config.Workflows, kind, body.Task, body.Recipe, body.Input, body.Position, body.Limit)
	if errors.Is(err, errWorkflowPreviewUnavailable) {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, err.Error())
		return
	}
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	writeJSON(response, http.StatusOK, preview)
}

// submitWorkflow runs the capability as its listing resolved it.
func (h *Handler) submitWorkflow(
	ctx context.Context,
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
						return h.config.Workflows.ExecuteWorkflow(withWorkflowSources(ctx, sources), kind, capability, input, reporter)
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
				control.Format != "" && control.Type != workflowControlDataset ||
				control.Bounds != nil && (control.Bounds.Step < 0 || control.Bounds.Rate < 0) {
				return fmt.Errorf("workflow workspace: invalid control %q", control.Name)
			}
			if control.Audio != nil && (control.Type != workflowControlArtifact || control.Media != "audio" || control.Audio.Format.Validate() != nil || control.Audio.MaximumEncodedBytes == 0 || control.Audio.MaximumSamples == 0) {
				return fmt.Errorf("workflow workspace: invalid audio input %q", control.Name)
			}
			fields[control.Name] = true
		}
	}
	return nil
}

func (kind workflowControlType) valid() bool {
	return kind == workflowControlText || kind == workflowControlInteger ||
		kind == workflowControlNumber || kind == workflowControlBoolean ||
		kind == workflowControlDataset || kind == workflowControlCheckpoint ||
		kind == workflowControlOutput || kind == workflowControlArtifact
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

func validateWorkflowInput(controls []workflowControl, raw json.RawMessage) error {
	fields := make(map[string]json.RawMessage, len(controls))
	if err := strictjson.Decode(bytes.NewReader(raw), &fields); err != nil {
		return fmt.Errorf("workflow workspace: invalid input: %w", err)
	}
	declarations := make(map[string]workflowControl, len(controls))
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

func validateWorkflowValue(controlType workflowControlType, raw json.RawMessage) error {
	var target any
	switch controlType {
	case workflowControlText, workflowControlOutput:
		target = new(string)
	case workflowControlArtifact:
		// The page fills an artifact control from a media card's stored id;
		// a value that is no artifact id is refused before any run.
		var value string
		if err := strictjson.Decode(bytes.NewReader(raw), &value); err != nil {
			return err
		}
		_, err := artifact.ParseID(value)
		return err
	case workflowControlDataset, workflowControlCheckpoint:
		var value string
		if err := strictjson.Decode(bytes.NewReader(raw), &value); err != nil {
			return err
		}
		id, err := artifact.ParseID(value)
		artifactKind := artifact.KindDataset
		if controlType == workflowControlCheckpoint {
			artifactKind = artifact.KindCheckpoint
		}
		if err != nil || id.Kind() != artifactKind {
			return fmt.Errorf("%s artifact required", artifactKind)
		}
		return nil
	case workflowControlInteger:
		target = new(int64)
	case workflowControlNumber:
		target = new(float64)
	case workflowControlBoolean:
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
