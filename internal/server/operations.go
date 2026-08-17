package server

import (
	"net/http"

	"overgo/internal/artifact"
)

type operationCancelRequest struct {
	ID artifact.ID `json:"id"`
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
