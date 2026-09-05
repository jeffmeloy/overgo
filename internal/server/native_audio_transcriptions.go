package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
)

var errTranscriptionUploadLimit = errors.New("transcription upload exceeds the request limit")

func (h *Handler) nativeAudioTranscriptions(response http.ResponseWriter, request *http.Request) {
	workspace, ok := h.generator.(WorkflowWorkspaceAPI)
	if !ok || h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "transcription workspace is unavailable")
		return
	}
	data, model, err := readTranscriptionUpload(response, request)
	if err != nil {
		_, bodyLimit := errors.AsType[*http.MaxBytesError](err)
		if errors.Is(err, errTranscriptionUploadLimit) || bodyLimit {
			writeError(response, http.StatusRequestEntityTooLarge, "invalid_audio", err.Error())
			return
		}
		writeInvalidRequest(response, err)
		return
	}
	capabilities, err := workspace.WorkflowCapabilities(request.Context(), WorkflowGeneration)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	capability, err := selectNativeWorkflowCapability(capabilities, recipe.TaskTranscription, model)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	descriptor, found, err := h.repository.Artifact(request.Context(), id)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	if !found {
		descriptor = artifact.Descriptor{ID: id, Size: uint64(len(data))}
	}
	_, err = artifact.CommitBatch(request.Context(), h.repository, artifact.Batch{
		Key: "transcription/upload/" + id.String(), Contents: []artifact.Content{{Descriptor: descriptor, Data: data}},
	})
	if err != nil && !errors.Is(err, artifact.ErrNoChange) {
		writeGenerationError(response, err)
		return
	}
	input, err := json.Marshal(transcriptionWorkflowInput{Audio: id})
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	status, err := h.waitNativeWorkflow(request.Context(), workspace, capability, input)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	if status.State != operation.StateCompleted || len(status.Outputs) != 1 {
		if status.Run != nil {
			run, readErr := runrecord.RequireExactRun(request.Context(), h.repository, *status.Run)
			if readErr == nil && run.Failure == transcriptionUploadLimit {
				writeError(response, http.StatusRequestEntityTooLarge, "invalid_audio", "audio exceeds the recipe's encoded-byte bound")
				return
			}
			if readErr == nil && (run.Failure == speechrecognition.AudioAdmissionFailure || run.Failure == speechrecognition.AudioFormatFailure) {
				writeInvalidRequestMessage(response, "audio does not satisfy the transcription recipe's admission policy")
				return
			}
		}
		writeError(response, http.StatusInternalServerError, "transcription_failed", status.Failure)
		return
	}
	transcription, err := speechrecognition.RequireTranscription(request.Context(), h.repository, status.Outputs[0])
	if err != nil {
		writeError(response, http.StatusInternalServerError, "invalid_output", "transcription output is unavailable or invalid")
		return
	}
	writeJSON(response, http.StatusOK, struct {
		Text string `json:"text"`
	}{Text: transcription.Text})
}

func readTranscriptionUpload(response http.ResponseWriter, request *http.Request) ([]byte, string, error) {
	request.Body = http.MaxBytesReader(response, request.Body, maxMultimodalRequestBytes)
	reader, err := request.MultipartReader()
	if err != nil {
		return nil, "", err
	}
	fields := make(map[string]string)
	var audio []byte
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, "", err
		}
		name := part.FormName()
		limit := maxRequestBytes
		switch name {
		case "file":
			if audio != nil || part.FileName() == "" {
				return nil, "", errors.New("one file upload is required")
			}
			limit = maxMediaBytes
		case "model", "response_format", "stream":
			if _, duplicate := fields[name]; duplicate || part.FileName() != "" {
				return nil, "", errors.New("duplicate or invalid transcription field")
			}
		default:
			return nil, "", errors.New("unsupported transcription field: " + name)
		}
		data, err := io.ReadAll(io.LimitReader(part, int64(limit)+1))
		if err != nil {
			return nil, "", err
		}
		if len(data) > limit {
			return nil, "", errTranscriptionUploadLimit
		}
		if name == "file" {
			if len(data) == 0 {
				return nil, "", errors.New("audio file is empty")
			}
			audio = data
		} else {
			fields[name] = string(data)
		}
	}
	if len(audio) == 0 || fields["model"] == "" {
		return nil, "", errors.New("file and model are required")
	}
	if format, supplied := fields["response_format"]; supplied && format != "json" {
		return nil, "", errors.New("response_format must be json")
	}
	if stream, supplied := fields["stream"]; supplied && stream != "false" {
		return nil, "", errors.New("streaming transcription is not supported")
	}
	return audio, fields["model"], request.Context().Err()
}
