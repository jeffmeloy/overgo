package server

import "net/http"

// requireMethod: exact endpoint method
func requireMethod(response http.ResponseWriter, request *http.Request, method string) bool {
	if request.Method == method {
		return true
	}
	response.Header().Set("Allow", method)
	writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", method+" required")
	return false
}

// requireModel: optional model selector check.
func (h *Handler) requireModel(response http.ResponseWriter, requested string) bool {
	if requested == "" || requested == h.config.ModelID {
		return true
	}
	writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
	return false
}
