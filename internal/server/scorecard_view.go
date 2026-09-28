package server

import (
	"context"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// ScorecardRow is one activated model and task on the user scorecard, its
// cells in the order a user meets them.
type ScorecardRow struct {
	Model  artifact.ID     `json:"model"`
	Name   string          `json:"name"`
	Task   string          `json:"task"`
	Recipe artifact.ID     `json:"recipe,omitzero"`
	Cells  []ScorecardCell `json:"cells"`
}

// ScorecardCell is one metric's measured value with its unit, what it
// measures and the record it came from, or the reason no record measures it.
type ScorecardCell struct {
	Metric string      `json:"metric"`
	Value  float64     `json:"value"`
	Unit   string      `json:"unit,omitzero"`
	Label  string      `json:"label,omitzero"`
	Source artifact.ID `json:"source,omitzero"`
	Reason string      `json:"reason,omitzero"`
}

// ScorecardReader reads the user scorecard from the store.
type ScorecardReader func(context.Context, *overgodb.Store) ([]ScorecardRow, error)

// scorecardResponse is the scorecard as the Library tab reads it.
type scorecardResponse struct {
	Rows []ScorecardRow `json:"rows"`
}

// scorecardView answers GET /scorecard with the user scorecard read from the
// store.
func (h *Handler) scorecardView(response http.ResponseWriter, request *http.Request) {
	if h.config.Scorecard == nil {
		writeError(response, http.StatusNotFound, "not_found", "the scorecard needs the launcher's scorecard reader")
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusServiceUnavailable, "repository_unavailable", "the scorecard reads the OvergoDB store, and this server has none")
		return
	}
	rows, err := h.config.Scorecard(request.Context(), h.repository)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "scorecard_unreadable", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, scorecardResponse{Rows: append([]ScorecardRow{}, rows...)})
}
