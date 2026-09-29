package server

import "net/http"

// agentRoute serves a route only when the agent runtime is configured (its
// coordinator binds the served store and interaction identity).
func agentRoute(serve routeHandler) routeHandler {
	return func(h *Handler, response http.ResponseWriter, request *http.Request) {
		if h.agentCoordinator == nil {
			writeError(response, http.StatusServiceUnavailable, "agent_unavailable", "no agent runtime is configured")
			return
		}
		serve(h, response, request)
	}
}
