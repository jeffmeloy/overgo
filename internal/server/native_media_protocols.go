package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"overgo/internal/artifact"
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
	store, release, ok := h.openBrowseStore(response)
	if !ok {
		return
	}
	defer release()
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
	store, release, ok := h.openBrowseStore(response)
	if !ok {
		return
	}
	defer release()
	descriptor, reader, found, err := store.OpenContent(request.Context(), status.Outputs[0])
	if err != nil || !found || !strings.HasPrefix(descriptor.MediaType, "audio/") {
		writeError(response, http.StatusInternalServerError, "invalid_output", "speech output is not an audio artifact")
		return
	}
	response.Header().Set("Content-Type", descriptor.MediaType)
	response.WriteHeader(http.StatusOK)
	_, _ = io.Copy(response, reader)
}

func (h *Handler) nativeWorkflowRequest(
	response http.ResponseWriter,
	request *http.Request,
	task recipe.Task,
) (WorkflowWorkspaceAPI, WorkflowCapability, map[string]json.RawMessage, bool) {
	if !requireMethod(response, request, http.MethodPost) {
		return nil, WorkflowCapability{}, nil, false
	}
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
	capability, err := selectNativeWorkflowCapability(capabilities, task, model)
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

func selectNativeWorkflowCapability(
	capabilities []WorkflowCapability,
	task recipe.Task,
	model string,
) (WorkflowCapability, error) {
	if err := validateWorkflowCapabilities(capabilities); err != nil {
		return WorkflowCapability{}, err
	}
	for _, capability := range capabilities {
		if capability.Task == task && (model == "" || model == capability.Recipe.String()) {
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
	id, err := h.submitWorkflow(request.Context(), workspace, WorkflowGeneration, capability, input, artifact.ID{})
	if err != nil {
		writeGenerationError(response, err)
		return operation.Status{}, false
	}
	status, err := h.operations.Wait(request.Context(), id)
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
