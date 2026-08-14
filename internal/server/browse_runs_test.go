package server

import (
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

func TestShapeRunProjectsRecord(t *testing.T) {
	run := runrecord.Run{
		Outcome:    runrecord.OutcomeSucceeded,
		Failure:    "",
		CodeCommit: "abc123",
		MeasuredNS: 2_500_000_000, // 2500 ms
		Inputs:     make([]artifact.ID, 0),
		Outputs:    make([]artifact.ID, 2),
		Phases: []runrecord.PhaseMetric{
			{Phase: runrecord.PhaseDecode, DurationNS: 1_000_000}, // 1 ms
		},
	}
	entry := shapeRun(run)
	if entry.Outcome != "succeeded" || entry.CodeCommit != "abc123" {
		t.Fatalf("entry = %+v", entry)
	}
	if entry.MeasuredMS != 2500 {
		t.Fatalf("measured_ms = %v, want 2500", entry.MeasuredMS)
	}
	if entry.Outputs != 2 || entry.Inputs != 0 {
		t.Fatalf("inputs/outputs = %d/%d", entry.Inputs, entry.Outputs)
	}
	if len(entry.Phases) != 1 || entry.Phases[0].Phase != "decode" || entry.Phases[0].MS != 1 {
		t.Fatalf("phases = %+v", entry.Phases)
	}
}

func TestBrowseRunsUnconfigured(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{}) // no RepoDBPath
	response := serveTestRequest(handler, http.MethodGet, "/runs", "")
	if response.Code != http.StatusNotImplemented {
		t.Fatalf("status=%d, want 501", response.Code)
	}
}
