package server

import (
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/runrecord"
)

type interactionReplayResponse struct {
	ID             artifact.ID                            `json:"id"`
	Trace          runrecord.InteractionTrace             `json:"trace"`
	TerminalReason runrecord.InteractionTerminalReason    `json:"terminal_reason,omitzero"`
	Media          []interactionMediaRef                  `json:"media,omitempty"`
	Comparison     *evaluation.InteractionTraceComparison `json:"comparison,omitempty"`
}

// interactionMediaRef names one media artifact an interaction carries
// with the declared media type the GUI needs to render it inline --
// an image, video, or audio element over /artifacts/content instead
// of an opaque identity.
type interactionMediaRef struct {
	ID        artifact.ID `json:"id"`
	MediaType string      `json:"media_type,omitzero"`
}

func (h *Handler) interactionReplay(response http.ResponseWriter, request *http.Request) {
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
	result := interactionReplayResponse{ID: trace.ID, Trace: trace, TerminalReason: interaction.TerminalReason}
	for _, mediaID := range interaction.Media {
		reference := interactionMediaRef{ID: mediaID}
		if descriptor, present, err := h.repository.Artifact(request.Context(), mediaID); err == nil && present {
			reference.MediaType = descriptor.MediaType
		}
		result.Media = append(result.Media, reference)
	}
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
