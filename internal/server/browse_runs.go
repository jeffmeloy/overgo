package server

import (
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

const (
	browseRunsDefaultLimit = 50
	browseRunsMaxLimit     = 500
)

type browseRunPhase struct {
	Phase string  `json:"phase"`
	MS    float64 `json:"ms"`
}

type browseRunEntry struct {
	ID         string           `json:"id"`
	Recipe     string           `json:"recipe"`
	Outcome    string           `json:"outcome"`
	Failure    string           `json:"failure,omitempty"`
	CodeCommit string           `json:"code_commit"`
	MeasuredMS float64          `json:"measured_ms"`
	Inputs     int              `json:"inputs"`
	Outputs    int              `json:"outputs"`
	Phases     []browseRunPhase `json:"phases"`
}

type browseRunsResponse struct {
	Count     int              `json:"count"`
	Offset    int              `json:"offset"`
	Limit     int              `json:"limit"`
	Truncated bool             `json:"truncated"`
	Runs      []browseRunEntry `json:"runs"`
}

// browseRuns: read-only listing of training/run artifacts from the RepoDB —
// outcome, recipe, code commit, wall time, and per-phase durations, decoded via
// runrecord.ParseRun. No job control. Opens the store read-only per request.
func (h *Handler) browseRuns(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if h.config.RepoDBPath == "" {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "run browsing is not configured")
		return
	}
	store, err := repodb.OpenReadOnly(h.config.RepoDBPath)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "repodb_error", "cannot open the run store")
		return
	}
	defer store.Close()
	// MaxResults must be a positive bound; fetch up to the store's cap so the
	// full set is pageable here. Truncated is surfaced if the cap is hit.
	result, err := store.Query(request.Context(), repodb.Query{Kind: artifact.KindRun, MaxResults: repodb.MaxQueryResults})
	if err != nil {
		writeError(response, http.StatusInternalServerError, "repodb_error", err.Error())
		return
	}

	descriptors := result.Artifacts
	total := len(descriptors)
	query := request.URL.Query()
	offset := clampNonNegative(parseIntDefault(query.Get("offset"), 0))
	if offset > total {
		offset = total
	}
	limit := parseIntDefault(query.Get("limit"), browseRunsDefaultLimit)
	if limit <= 0 {
		limit = browseRunsDefaultLimit
	}
	if limit > browseRunsMaxLimit {
		limit = browseRunsMaxLimit
	}
	end := offset + limit
	if end > total {
		end = total
	}

	runs := make([]browseRunEntry, 0, end-offset)
	for _, descriptor := range descriptors[offset:end] {
		content, ok, contentErr := store.Content(request.Context(), descriptor.ID)
		if contentErr != nil || !ok {
			continue
		}
		run, parseErr := runrecord.ParseRun(content.Data)
		if parseErr != nil {
			continue
		}
		runs = append(runs, shapeRun(run))
	}
	writeJSON(response, http.StatusOK, browseRunsResponse{Count: total, Offset: offset, Limit: limit, Truncated: result.Truncated, Runs: runs})
}

// shapeRun: a run record projected to the browse response (pure; nanoseconds
// rendered as milliseconds).
func shapeRun(run runrecord.Run) browseRunEntry {
	phases := make([]browseRunPhase, 0, len(run.Phases))
	for _, phase := range run.Phases {
		phases = append(phases, browseRunPhase{Phase: string(phase.Phase), MS: float64(phase.DurationNS) / 1e6})
	}
	return browseRunEntry{
		ID:         run.ID.String(),
		Recipe:     run.Recipe.String(),
		Outcome:    string(run.Outcome),
		Failure:    run.Failure,
		CodeCommit: run.CodeCommit,
		MeasuredMS: float64(run.MeasuredNS) / 1e6,
		Inputs:     len(run.Inputs),
		Outputs:    len(run.Outputs),
		Phases:     phases,
	}
}
