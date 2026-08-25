package server

import (
	"context"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
)

type peerCapabilityRequest struct {
	Peer            artifact.ID                      `json:"peer"`
	Capability      modelrecipe.RemotePeerCapability `json:"capability"`
	PublishedUnixNS int64                            `json:"published_unix_ns"`
}

type peerStateRequest struct {
	Peer          artifact.ID                       `json:"peer"`
	State         runrecord.PeerAdministrativeState `json:"state"`
	ChangedUnixNS int64                             `json:"changed_unix_ns"`
}

func (h *Handler) peerWorkspace(response http.ResponseWriter, request *http.Request) {
	workspace, ok := h.generator.(PeerWorkspaceAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "peer workspace is unavailable")
		return
	}
	switch request.URL.Path {
	case "/peers":
		if !requireMethod(response, request, http.MethodGet) {
			return
		}
		limit := boundedPeerProjectionLimit(request, h.config.MaxStoredResponses)
		value, err := workspace.PeerInventory(request.Context(), limit)
		writePeerResult(response, http.StatusOK, value, err)
	case "/peers/enroll":
		var body runrecord.PeerEnrollment
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		enrollment, state, err := workspace.EnrollPeer(request.Context(), body)
		writePeerResult(response, http.StatusCreated, struct {
			Peer       artifact.ID              `json:"peer"`
			Enrollment runrecord.PeerEnrollment `json:"enrollment"`
			StateID    artifact.ID              `json:"state_id"`
			State      runrecord.PeerState      `json:"state"`
		}{enrollment.ID, enrollment, state.ID, state}, err)
	case "/peers/capability":
		var body peerCapabilityRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		publication, capability, err := workspace.PublishPeerCapability(request.Context(), body.Peer, body.Capability, body.PublishedUnixNS)
		writePeerResult(response, http.StatusCreated, struct {
			PublicationID artifact.ID                           `json:"publication_id"`
			Publication   modelrecipe.PeerCapabilityPublication `json:"publication"`
			Capability    modelrecipe.RemotePeerCapability      `json:"capability"`
		}{publication.ID, publication, capability}, err)
	case "/peers/heartbeat":
		var body runrecord.PeerHeartbeat
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		value, err := workspace.HeartbeatPeer(request.Context(), body)
		writePeerResult(response, http.StatusCreated, struct {
			ID        artifact.ID             `json:"id"`
			Heartbeat runrecord.PeerHeartbeat `json:"heartbeat"`
		}{value.ID, value}, err)
	case "/peers/state":
		var body peerStateRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		value, err := workspace.TransitionPeer(request.Context(), body.Peer, body.State, body.ChangedUnixNS)
		writePeerResult(response, http.StatusOK, struct {
			ID    artifact.ID         `json:"id"`
			State runrecord.PeerState `json:"state"`
		}{value.ID, value}, err)
	case "/peers/placement":
		var body modelrecipe.PeerPlacementRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		value, err := workspace.CompilePeerPlacement(request.Context(), body)
		writePeerResult(response, http.StatusOK, value, err)
	case "/peers/reconcile":
		var body capabilityruntime.PeerReconcileRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		id, err := workspace.ReconcilePeer(context.WithoutCancel(request.Context()), h.operations, body)
		writePeerResult(response, http.StatusAccepted, struct {
			Operation artifact.ID `json:"operation"`
		}{id}, err)
	case "/peers/evidence":
		if !requireMethod(response, request, http.MethodGet) {
			return
		}
		id, err := artifact.ParseID(request.URL.Query().Get("operation"))
		if err != nil {
			writeError(response, http.StatusBadRequest, "invalid_operation", "peer operation identity is required")
			return
		}
		limit := boundedPeerProjectionLimit(request, h.config.MaxStoredResponses)
		value, evidenceErr := workspace.PeerEvidence(request.Context(), h.operations, id, limit)
		writePeerResult(response, http.StatusOK, value, evidenceErr)
	case "/peers/stream":
		h.peerStream(response, request, workspace)
	default:
		writeError(response, http.StatusNotFound, "not_found", "peer route is absent")
	}
}

func (h *Handler) peerStream(response http.ResponseWriter, request *http.Request, workspace PeerWorkspaceAPI) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	events, unsubscribe, err := h.operations.Subscribe()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	defer unsubscribe()
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	limit := boundedPeerProjectionLimit(request, h.config.MaxStoredResponses)
	inventory, err := workspace.PeerInventory(request.Context(), limit)
	if err != nil || stream.named("peer.inventory", inventory) != nil ||
		stream.named("operation.snapshot", h.operations.List()) != nil {
		return
	}
	for {
		select {
		case <-request.Context().Done():
			return
		case event, open := <-events:
			if !open || stream.named("operation", event) != nil {
				return
			}
		}
	}
}

func boundedPeerProjectionLimit(request *http.Request, maximum int) int {
	limit := parseIntDefault(request.URL.Query().Get("limit"), maximum)
	if limit <= 0 || limit > maximum {
		return maximum
	}
	return limit
}

func writePeerResult(response http.ResponseWriter, status int, value any, err error) {
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "peer_refused", err.Error())
		return
	}
	writeJSON(response, status, value)
}
