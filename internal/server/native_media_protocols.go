package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

type nativeImageData struct {
	URL string `json:"url"`
}

type nativeImageResponse struct {
	Created int64             `json:"created"`
	Data    []nativeImageData `json:"data"`
}

func (h *Handler) nativeImageGeneration(response http.ResponseWriter, request *http.Request) {
	started := time.Now()
	workspace, capability, fields, ok := h.nativeWorkflowRequest(response, request, recipe.TaskImageGen)
	if !ok {
		return
	}
	format, err := takeNativeString(fields, "response_format")
	if err != nil || format != "" && format != "url" {
		writeInvalidRequestMessage(response, "response_format must be url")
		return
	}
	input, err := marshalWorkflowInput(capability.Controls, fields)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	status, ok := h.runNativeWorkflow(response, request, workspace, capability, input)
	if !ok {
		return
	}
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	data := make([]nativeImageData, len(status.Outputs))
	for index, id := range status.Outputs {
		descriptor, found, err := store.Artifact(request.Context(), id)
		if err != nil || !found || !strings.HasPrefix(descriptor.MediaType, "image/") {
			writeError(response, http.StatusInternalServerError, "invalid_output", "generation output is not an image artifact")
			return
		}
		data[index].URL = "/artifacts/content?id=" + url.QueryEscape(id.String())
	}
	writeJSON(response, http.StatusOK, nativeImageResponse{Created: started.Unix(), Data: data})
}

// nativeVideoGeneration runs the registered text-to-video capability
// through the same native workflow the image route uses: the prompt
// and controls become one workflow operation, the outputs must be
// committed video artifacts, and the response carries their content
// URLs so the workbench plays them straight from the store.
func (h *Handler) nativeVideoGeneration(response http.ResponseWriter, request *http.Request) {
	started := time.Now()
	workspace, capability, fields, ok := h.nativeWorkflowRequest(response, request, recipe.TaskVideoGen)
	if !ok {
		return
	}
	input, err := marshalWorkflowInput(capability.Controls, fields)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	status, ok := h.runNativeWorkflow(response, request, workspace, capability, input)
	if !ok {
		return
	}
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	data := make([]nativeImageData, len(status.Outputs))
	for index, id := range status.Outputs {
		descriptor, found, err := store.Artifact(request.Context(), id)
		if err != nil || !found || !strings.HasPrefix(descriptor.MediaType, "video/") {
			writeError(response, http.StatusInternalServerError, "invalid_output", "generation output is not a video artifact")
			return
		}
		data[index].URL = "/artifacts/content?id=" + url.QueryEscape(id.String())
	}
	writeJSON(response, http.StatusOK, nativeImageResponse{Created: started.Unix(), Data: data})
}

// nativeVideoEdit runs the registered reference-guided video-editing
// capability: a source video and a prompt become one workflow
// operation over the reference-edit runtime, and the edited output
// must be a committed video artifact served by its content URL.
func (h *Handler) nativeVideoEdit(response http.ResponseWriter, request *http.Request) {
	started := time.Now()
	workspace, capability, fields, ok := h.nativeWorkflowRequest(response, request, recipe.TaskVideoEdit)
	if !ok {
		return
	}
	input, err := marshalWorkflowInput(capability.Controls, fields)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	status, ok := h.runNativeWorkflow(response, request, workspace, capability, input)
	if !ok {
		return
	}
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	data := make([]nativeImageData, len(status.Outputs))
	for index, id := range status.Outputs {
		descriptor, found, err := store.Artifact(request.Context(), id)
		if err != nil || !found || !strings.HasPrefix(descriptor.MediaType, "video/") {
			writeError(response, http.StatusInternalServerError, "invalid_output", "edit output is not a video artifact")
			return
		}
		data[index].URL = "/artifacts/content?id=" + url.QueryEscape(id.String())
	}
	writeJSON(response, http.StatusOK, nativeImageResponse{Created: started.Unix(), Data: data})
}

func (h *Handler) nativeAudioSpeech(response http.ResponseWriter, request *http.Request) {
	workspace, capability, fields, ok := h.nativeWorkflowRequest(response, request, recipe.TaskSpeech)
	if !ok {
		return
	}
	input, err := marshalWorkflowInput(capability.Controls, fields)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	status, ok := h.runNativeWorkflow(response, request, workspace, capability, input)
	if !ok {
		return
	}
	if len(status.Outputs) != 1 {
		writeError(response, http.StatusInternalServerError, "invalid_output", "speech generation requires one audio artifact")
		return
	}
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	descriptor, reader, found, err := store.OpenContent(request.Context(), status.Outputs[0])
	if err != nil || !found || !strings.HasPrefix(descriptor.MediaType, "audio/") {
		if closer, ok := reader.(io.Closer); ok {
			_ = closer.Close()
		}
		writeError(response, http.StatusInternalServerError, "invalid_output", "speech output is not an audio artifact")
		return
	}
	response.Header().Set("X-Overgo-Artifact", status.Outputs[0].String()) // the audio artifact, for provenance
	response.Header().Set("Content-Type", descriptor.MediaType)
	response.WriteHeader(http.StatusOK)
	_, _ = io.Copy(response, reader)
}

func (h *Handler) nativeWorkflowRequest(
	response http.ResponseWriter,
	request *http.Request,
	task recipe.Task,
) (WorkflowWorkspaceAPI, WorkflowCapability, map[string]json.RawMessage, bool) {
	workspace, ok := h.generator.(WorkflowWorkspaceAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "generation workspace is unavailable")
		return nil, WorkflowCapability{}, nil, false
	}
	fields := make(map[string]json.RawMessage)
	if !h.decodeBoundedJSON(response, request, &fields) {
		return nil, WorkflowCapability{}, nil, false
	}
	model, err := takeNativeString(fields, "model")
	if err != nil {
		writeInvalidRequest(response, err)
		return nil, WorkflowCapability{}, nil, false
	}
	capabilities, err := workspace.WorkflowCapabilities(request.Context(), WorkflowGeneration)
	if err != nil {
		writeGenerationError(response, err)
		return nil, WorkflowCapability{}, nil, false
	}
	capability, err := h.selectNativeWorkflowCapability(capabilities, task, model)
	if err != nil {
		writeInvalidRequest(response, err)
		return nil, WorkflowCapability{}, nil, false
	}
	return workspace, capability, fields, true
}

func marshalWorkflowInput(controls []WorkflowControl, fields map[string]json.RawMessage) (json.RawMessage, error) {
	input, err := json.Marshal(fields)
	if err == nil {
		err = validateWorkflowInput(controls, input)
	}
	return input, err
}

func (h *Handler) selectNativeWorkflowCapability(
	capabilities []WorkflowCapability,
	task recipe.Task,
	model string,
) (WorkflowCapability, error) {
	if err := validateWorkflowCapabilities(capabilities); err != nil {
		return WorkflowCapability{}, err
	}
	if model == h.config.ModelID && h.modelArtifact.Valid() {
		model = h.modelArtifact.String()
	}
	for _, capability := range capabilities {
		if capability.Task == task && capability.Refusal == "" && (model == "" || model == capability.Recipe.String() || model == capability.model.String()) {
			return capability, nil
		}
	}
	return WorkflowCapability{}, errors.New("workflow workspace: active native media recipe is absent")
}

func (h *Handler) runNativeWorkflow(
	response http.ResponseWriter,
	request *http.Request,
	workspace WorkflowWorkspaceAPI,
	capability WorkflowCapability,
	input json.RawMessage,
) (operation.Status, bool) {
	status, err := h.waitNativeWorkflow(request.Context(), workspace, capability, input)
	if err != nil {
		writeGenerationError(response, err)
		return operation.Status{}, false
	}
	if status.State != operation.StateCompleted || len(status.Outputs) == 0 {
		writeError(response, http.StatusInternalServerError, "generation_failed", status.Failure)
		return operation.Status{}, false
	}
	return status, true
}

func (h *Handler) waitNativeWorkflow(ctx context.Context, workspace WorkflowWorkspaceAPI, capability WorkflowCapability, input json.RawMessage) (operation.Status, error) {
	id, err := h.submitWorkflow(ctx, workspace, WorkflowGeneration, capability, input, nil)
	if err != nil {
		return operation.Status{}, err
	}
	return h.operations.Wait(ctx, id)
}

func takeNativeString(fields map[string]json.RawMessage, name string) (string, error) {
	raw, ok := fields[name]
	if !ok {
		return "", nil
	}
	delete(fields, name)
	var value string
	if err := strictjson.DecodeBytes(raw, &value); err != nil {
		return "", err
	}
	return value, nil
}
