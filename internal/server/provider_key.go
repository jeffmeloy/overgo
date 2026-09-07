package server

import (
	"net/http"
)

// providerKeyRequest: the key for the hosted model a reference names.
type providerKeyRequest struct {
	Location string `json:"location"`
	Key      string `json:"key"`
}

// providerKeyResponse: the variable now holding the key.
type providerKeyResponse struct {
	Location       string `json:"location"`
	KeyEnvironment string `json:"key_environment"`
}

// providerKey places a hosted provider's key in this process through the
// launcher's intake; the key is never written to the store or a log. The
// catalog lists the model servable from the next listing.
func (h *Handler) providerKey(response http.ResponseWriter, request *http.Request) {
	var body providerKeyRequest
	if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if h.config.Repository == nil || h.config.ProviderKeys == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "provider keys need a store and the launcher's provider intake")
		return
	}
	variable, err := h.config.ProviderKeys(request.Context(), h.config.Repository, body.Location, body.Key)
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "provider_refused", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, providerKeyResponse{Location: body.Location, KeyEnvironment: variable})
}
