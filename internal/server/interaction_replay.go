package server

import (
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/runrecord"
)

type interactionReplayResponse struct {
	ID         artifact.ID                            `json:"id"`
	Trace      runrecord.InteractionTrace             `json:"trace"`
	Comparison *evaluation.InteractionTraceComparison `json:"comparison,omitempty"`
}

func (h *Handler) interactionReplay(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "interaction repository is unavailable")
		return
	}
	interaction, found, err := runrecord.ResolveInteraction(request.Context(), h.repository, request.URL.Query().Get("response"))
	if err != nil || !found {
		writeError(response, http.StatusNotFound, "not_found", "interaction trace not found")
		return
	}
	trace, err := runrecord.RequireInteractionTrace(request.Context(), h.repository, interaction.Trace)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	result := interactionReplayResponse{ID: trace.ID, Trace: trace}
	if value := request.URL.Query().Get("baseline"); value != "" {
		baselineID, parseErr := artifact.ParseID(value)
		if parseErr != nil {
			writeInvalidRequest(response, parseErr)
			return
		}
		baseline, loadErr := runrecord.RequireInteractionTrace(request.Context(), h.repository, baselineID)
		if loadErr != nil {
			writeInvalidRequest(response, loadErr)
			return
		}
		comparison, compareErr := evaluation.CompareInteractionTraces(baseline, trace)
		if compareErr != nil {
			writeInvalidRequest(response, compareErr)
			return
		}
		result.Comparison = &comparison
	}
	writeJSON(response, http.StatusOK, result)
}
