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

// ScorecardMove is one thing that moved between the two newest published
// scorecards: a row that came or went, or one of its cells changed, gained
// or lost its measure. Improved says a changed cell moved the way a user
// wants.
type ScorecardMove struct {
	Name     string  `json:"name"`
	Task     string  `json:"task"`
	Kind     string  `json:"kind"`
	Metric   string  `json:"metric,omitzero"`
	Before   float64 `json:"before,omitzero"`
	After    float64 `json:"after,omitzero"`
	Unit     string  `json:"unit,omitzero"`
	Reason   string  `json:"reason,omitzero"`
	Improved bool    `json:"improved,omitzero"`
}

// ScorecardHistory is what moved from the previous published scorecard to
// the newest, with the commits each measured.
type ScorecardHistory struct {
	Previous string          `json:"previous"`
	Current  string          `json:"current"`
	Moves    []ScorecardMove `json:"moves"`
}

// ScorecardHistoryReader reads what moved between the two newest published
// scorecards; nil until two are published.
type ScorecardHistoryReader func() (*ScorecardHistory, error)

// scorecardResponse is the scorecard as the Library tab reads it, with what
// moved since the previous published one when there is one.
type scorecardResponse struct {
	Rows    []ScorecardRow    `json:"rows"`
	History *ScorecardHistory `json:"history,omitempty"`
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
	answer := scorecardResponse{Rows: append([]ScorecardRow{}, rows...)}
	if h.config.ScorecardHistory != nil {
		if answer.History, err = h.config.ScorecardHistory(); err != nil {
			writeError(response, http.StatusInternalServerError, "scorecard_unreadable", err.Error())
			return
		}
	}
	writeJSON(response, http.StatusOK, answer)
}
