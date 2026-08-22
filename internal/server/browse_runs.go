package server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"slices"

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

type browseRunDetail struct {
	ID          artifact.ID             `json:"id"`
	Recipe      artifact.ID             `json:"recipe"`
	Outcome     runrecord.Outcome       `json:"outcome"`
	Failure     string                  `json:"failure,omitempty"`
	CodeCommit  string                  `json:"code_commit,omitempty"`
	Environment *artifact.ID            `json:"environment,omitempty"`
	MeasuredNS  uint64                  `json:"measured_ns,omitempty"`
	Inputs      []artifact.ID           `json:"inputs,omitempty"`
	Outputs     []artifact.ID           `json:"outputs,omitempty"`
	Phases      []runrecord.PhaseMetric `json:"phases,omitempty"`
	Parents     []artifact.Lineage      `json:"parents"`
	Children    []artifact.Lineage      `json:"children"`
}

// browseRuns: read-only listing of training/run artifacts from the RepoDB —
// outcome, recipe, code commit, wall time, and per-phase durations, decoded via
// runrecord.ParseRun. No job control. Opens the store read-only per request.
func (h *Handler) browseRuns(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	store, release, ok := h.openBrowseStore(response)
	if !ok {
		return
	}
	defer release()
	if value := request.URL.Query().Get("id"); value != "" {
		id, parseErr := artifact.ParseID(value)
		if parseErr != nil || id.Kind() != artifact.KindRun {
			writeInvalidRequestMessage(response, "invalid run identity")
			return
		}
		detail, found, detailErr := loadRunDetail(request.Context(), store, id)
		if detailErr != nil {
			writeError(response, http.StatusInternalServerError, "repodb_error", detailErr.Error())
			return
		}
		if !found {
			writeError(response, http.StatusNotFound, "not_found", "run not found")
			return
		}
		writeJSON(response, http.StatusOK, detail)
		return
	}
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
		run, parseErr := runrecord.RequireRun(request.Context(), store, descriptor.ID)
		if parseErr != nil {
			continue
		}
		runs = append(runs, shapeRun(run))
	}
	writeJSON(response, http.StatusOK, browseRunsResponse{Count: total, Offset: offset, Limit: limit, Truncated: result.Truncated, Runs: runs})
}

func loadRunDetail(ctx context.Context, store *repodb.Store, id artifact.ID) (browseRunDetail, bool, error) {
	descriptor, reader, found, err := store.OpenContent(ctx, id)
	if err != nil || !found {
		return browseRunDetail{}, found, err
	}
	data, err := io.ReadAll(io.LimitReader(reader, int64(descriptor.Size)+1))
	if err != nil || uint64(len(data)) != descriptor.Size {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return browseRunDetail{}, false, err
	}
	run, err := runrecord.ParseRun(data)
	if err != nil {
		return browseRunDetail{}, false, err
	}
	if run.ID != id {
		return browseRunDetail{}, false, errors.New("run content identity differs")
	}
	parents, err := store.Parents(ctx, id)
	if err != nil {
		return browseRunDetail{}, false, err
	}
	children, err := store.Children(ctx, id)
	if err != nil {
		return browseRunDetail{}, false, err
	}
	for _, edge := range run.Lineage() {
		if edge.Child == id && !slices.Contains(parents, edge) || edge.Parent == id && !slices.Contains(children, edge) {
			return browseRunDetail{}, false, errors.New("run lineage is incomplete")
		}
	}
	detail := browseRunDetail{
		ID: run.ID, Recipe: run.Recipe, Outcome: run.Outcome, Failure: run.Failure,
		CodeCommit: run.CodeCommit, MeasuredNS: run.MeasuredNS,
		Inputs: run.Inputs, Outputs: run.Outputs, Phases: run.Phases,
		Parents: parents, Children: children,
	}
	if run.Environment.Valid() {
		detail.Environment = &run.Environment
	}
	return detail, true, nil
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
