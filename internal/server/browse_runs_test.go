package server

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
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

func TestBrowseRunsReadsRepoDBRecords(t *testing.T) {
	root := t.TempDir()
	store, err := repodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "browse-recipe")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "browse-output")
	if _, err := store.Commit(context.Background(), artifact.Batch{Key: "fixture/facts", Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: outputID}}}); err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewRun(recipeID, runrecord.OutcomeSucceeded, nil, []artifact.ID{outputID}, "")
	if err != nil {
		t.Fatal(err)
	}
	batch, err := run.Batch("fixture/run")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, RepoDBPath: root}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	response := serveTestRequest(handler, http.MethodGet, "/runs?limit=1", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result browseRunsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Count != 1 || len(result.Runs) != 1 || result.Runs[0].ID != run.ID.String() {
		t.Fatalf("runs=%+v", result)
	}
}
