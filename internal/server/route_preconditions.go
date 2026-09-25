package server

import "net/http"

// generatorWorkspace serves a route over a workspace the served generator
// may offer; one without it answers 501 with what is missing.
func generatorWorkspace[W any](absent string, serve func(*Handler, W, http.ResponseWriter, *http.Request)) routeHandler {
	return func(h *Handler, response http.ResponseWriter, request *http.Request) {
		workspace, ok := h.generator.(W)
		if !ok {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, absent)
			return
		}
		serve(h, workspace, response, request)
	}
}

func peerRoute(serve func(*Handler, PeerWorkspaceAPI, http.ResponseWriter, *http.Request)) routeHandler {
	return generatorWorkspace("peer workspace is unavailable", serve)
}

func automationRoute(serve func(*Handler, AutomationWorkspaceAPI, http.ResponseWriter, *http.Request)) routeHandler {
	return generatorWorkspace("automation workspace is unavailable", serve)
}

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
