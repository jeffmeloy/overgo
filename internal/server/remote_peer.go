package server

import (
	"net/http"

	"overgo/internal/modelrecipe"
)

type remotePeerAuthorityRequest struct {
	Capability    modelrecipe.RemotePeerCapability    `json:"capability"`
	Compatibility modelrecipe.RemotePeerCompatibility `json:"compatibility"`
}

func (h *Handler) remotePeerAuthority(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "remote peer repository is unavailable")
		return
	}
	var body remotePeerAuthorityRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	compatibility, err := modelrecipe.PublishRemotePeerAuthority(
		request.Context(), h.repository, "remote-peer/authority", body.Capability, body.Compatibility,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	writeJSON(response, http.StatusCreated, compatibility)
}
