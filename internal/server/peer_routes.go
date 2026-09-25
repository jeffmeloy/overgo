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

// The peer routes: each serves one path of the route table over the peer workspace (peerRoute).

func (h *Handler) peerInventory(workspace PeerWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	value, err := workspace.PeerInventory(request.Context(), boundedPeerProjectionLimit(request, h.config.MaxStoredResponses))
	writePeerResult(response, http.StatusOK, value, err)
}

func (h *Handler) peerEnroll(workspace PeerWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body runrecord.PeerEnrollment
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	enrollment, state, err := workspace.EnrollPeer(request.Context(), body)
	writePeerResult(response, http.StatusCreated, struct {
		Peer       artifact.ID              `json:"peer"`
		Enrollment runrecord.PeerEnrollment `json:"enrollment"`
		StateID    artifact.ID              `json:"state_id"`
		State      runrecord.PeerState      `json:"state"`
	}{enrollment.ID, enrollment, state.ID, state}, err)
}

func (h *Handler) peerCapability(workspace PeerWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body peerCapabilityRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	publication, capability, err := workspace.PublishPeerCapability(request.Context(), body.Peer, body.Capability, body.PublishedUnixNS)
	writePeerResult(response, http.StatusCreated, struct {
		PublicationID artifact.ID                           `json:"publication_id"`
		Publication   modelrecipe.PeerCapabilityPublication `json:"publication"`
		Capability    modelrecipe.RemotePeerCapability      `json:"capability"`
	}{publication.ID, publication, capability}, err)
}

func (h *Handler) peerHeartbeat(workspace PeerWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body runrecord.PeerHeartbeat
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	value, err := workspace.HeartbeatPeer(request.Context(), body)
	writePeerResult(response, http.StatusCreated, struct {
		ID        artifact.ID             `json:"id"`
		Heartbeat runrecord.PeerHeartbeat `json:"heartbeat"`
	}{value.ID, value}, err)
}

func (h *Handler) peerState(workspace PeerWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body peerStateRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	value, err := workspace.TransitionPeer(request.Context(), body.Peer, body.State, body.ChangedUnixNS)
	writePeerResult(response, http.StatusOK, struct {
		ID    artifact.ID         `json:"id"`
		State runrecord.PeerState `json:"state"`
	}{value.ID, value}, err)
}

func (h *Handler) peerPlacement(workspace PeerWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body modelrecipe.PeerPlacementRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	value, err := workspace.CompilePeerPlacement(request.Context(), body)
	writePeerResult(response, http.StatusOK, value, err)
}

func (h *Handler) peerReconcile(workspace PeerWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	var body capabilityruntime.PeerReconcileRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	id, err := workspace.ReconcilePeer(context.WithoutCancel(request.Context()), h.operations, body)
	writePeerResult(response, http.StatusAccepted, struct {
		Operation artifact.ID `json:"operation"`
	}{id}, err)
}

func (h *Handler) peerEvidence(workspace PeerWorkspaceAPI, response http.ResponseWriter, request *http.Request) {
	id, err := artifact.ParseID(request.URL.Query().Get("operation"))
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_operation", "peer operation identity is required")
		return
	}
	value, evidenceErr := workspace.PeerEvidence(request.Context(), h.operations, id, boundedPeerProjectionLimit(request, h.config.MaxStoredResponses))
	writePeerResult(response, http.StatusOK, value, evidenceErr)
}

func boundedPeerProjectionLimit(request *http.Request, maximum int) int {
	limit := parseIntDefault(request.URL.Query().Get("limit"), maximum)
	if limit <= 0 || limit > maximum {
		return maximum
	}
	return limit
}

func writePeerResult(response http.ResponseWriter, status int, value any, err error) {
	writeRefusableResult(response, status, value, err, "peer_refused")
}
