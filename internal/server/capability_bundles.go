package server

import (
	"net/http"

	"overgo/internal/artifact"
)

type capabilityBundleResponse struct {
	Bundles []artifact.ID     `json:"bundles"`
	Content *artifact.Content `json:"content,omitempty"`
}

func (h *Handler) capabilityBundles(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	loader, ok := h.tools.(capabilityBundleAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "capability bundles are unavailable")
		return
	}
	bundles := loader.CapabilityBundles()
	name := request.URL.Query().Get("name")
	if name == "" {
		writeJSON(response, http.StatusOK, capabilityBundleResponse{Bundles: bundles})
		return
	}
	bundle, err := artifact.ParseID(request.URL.Query().Get("bundle"))
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	role, err := artifact.ParseComponentRole(request.URL.Query().Get("role"))
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	content, err := loader.LoadCapabilityComponent(request.Context(), bundle, role, name)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	writeJSON(response, http.StatusOK, capabilityBundleResponse{Bundles: bundles, Content: &content})
}
