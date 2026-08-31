package server

import (
	"encoding/hex"
	"net/http"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

const (
	storeDeltasDefaultWindow = 64
	storeDeltasMaxWindow     = 512
)

// storeDeltasResponse serves one head-bound reconciliation step to a
// read-only peer or the embedded UI: either the ordered coalesced delta from
// the consumer's exact head, or a resync directive naming the current
// snapshot coordinate when no delta can honestly bridge.
type storeDeltasResponse struct {
	Resync            bool                     `json:"resync,omitzero"`
	Head              string                   `json:"head"`
	Sequence          uint64                   `json:"sequence"`
	ProjectionVersion string                   `json:"projection_version"`
	Delta             *overgodb.HeadBoundDelta `json:"delta,omitempty"`
}

// storeDeltas lets a stateful read-only consumer stop reloading complete
// state: it names its last reconciled head, sequence, and projection
// contract, and receives only the committed change since — or a resync
// directive when continuity or the projection contract broke.
func (h *Handler) storeDeltas(response http.ResponseWriter, request *http.Request) {
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	query := request.URL.Query()
	window := storeDeltasDefaultWindow
	if value := query.Get("limit"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 || parsed > storeDeltasMaxWindow {
			writeInvalidRequestMessage(response, "invalid delta window")
			return
		}
		window = parsed
	}
	var previousHead artifact.CommitID
	if value := query.Get("head"); value != "" {
		decoded, err := hex.DecodeString(value)
		if err != nil || len(decoded) != len(previousHead) {
			writeInvalidRequestMessage(response, "invalid consumer head")
			return
		}
		copy(previousHead[:], decoded)
	}
	var previousSequence uint64
	if value := query.Get("sequence"); value != "" {
		parsed, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			writeInvalidRequestMessage(response, "invalid consumer sequence")
			return
		}
		previousSequence = parsed
	}
	delta, resync, err := store.DeltasSince(
		request.Context(), previousHead, previousSequence, query.Get("projection"), window,
	)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	head, sequence := store.Head()
	result := storeDeltasResponse{
		Head: head.String(), Sequence: sequence,
		ProjectionVersion: overgodb.ProjectionContractVersion(),
	}
	if resync {
		result.Resync = true
		writeJSON(response, http.StatusOK, result)
		return
	}
	result.Delta = &delta
	writeJSON(response, http.StatusOK, result)
}
