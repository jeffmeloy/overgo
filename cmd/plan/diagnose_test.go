package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// seedFailedAttempt commits one failed gate attempt for the row with a
// failed test step carrying evidence.
func seedFailedAttempt(t *testing.T, root, item, step, evidence string) {
	t.Helper()
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	steps := []runrecord.GateStep{
		{Name: "protection", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepSucceeded, DurationNS: 5},
		{Name: "test", Phase: runrecord.PhaseTest, Outcome: runrecord.StepFailed, DurationNS: 9, Evidence: evidence},
	}
	record, err := runrecord.NewGateRecord(
		testutil.ArtifactID(t, artifact.KindRecipe, "diagnose recipe"),
		testutil.ArtifactID(t, artifact.KindEvidence, "diagnose environment"),
		strings.Repeat("cd", 20), runrecord.OutcomeFailed, "test", 14, steps,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch("diagnose/gate/" + record.Result.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: item, PlanStep: step, Result: record.Result.ID, Recipe: record.Result.Recipe,
		CodeCommit: strings.Repeat("cd", 20), Outcome: runrecord.OutcomeFailed, Failure: "test", WallNS: 14,
		Selection: runrecord.AttemptSelection{Defined: 2, Selected: 2}, Diff: runrecord.AttemptDiff{Files: 1, Insertions: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := attempt.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: record.Result.Recipe}, artifact.Descriptor{ID: record.Result.Environment}, content.Descriptor)
	batch.Contents = append(batch.Contents, content)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
}

// TestDiagnoseRowReference pins: a ready row with a failed attempt prints
// its state, the attempt, the failed step's bounded evidence and a fix
// action naming the step; a held row names what it waits on; a ready row
// without attempts is told to dispatch; an unknown row is absent.
func TestDiagnoseRowReference(t *testing.T) {
	document := plan.Plan{Items: []plan.Item{
		{ID: "guard", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "check", Status: plan.StatusOpen, Verify: "go test ./guard"}}},
		{ID: "dependent", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "pass", Status: plan.StatusOpen, Verify: "go test ./dependent", DependsOn: []string{"guard/check"},
		}}},
		{ID: "fresh", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "start", Status: plan.StatusOpen, Verify: "go test ./fresh"}}},
	}}
	root := initializePlanTestRepository(t, document)
	seedFailedAttempt(t, root, "guard", "check", "FAIL TestGuard guard_test.go:9: expected 1, got 2")
	var output bytes.Buffer
	if err := diagnoseRow(root, "guard/check", &output); err != nil {
		t.Fatal(err)
	}
	report := output.String()
	for _, want := range []string{
		"row guard/check: state=open-ready", "attempts: 1", "#1 failed:test wall=", "commit=cdcdcdcd",
		"failed step test (test):", "guard_test.go:9: expected 1, got 2", "gate failure: test",
		"next: fix the failed step(s) test and gate again: go run ./cmd/gate -plan guard/check",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("diagnosis lacks %q:\n%s", want, report)
		}
	}
	output.Reset()
	if err := diagnoseRow(root, "dependent/pass", &output); err != nil {
		t.Fatal(err)
	}
	if report := output.String(); !strings.Contains(report, "state=open-blocked") || !strings.Contains(report, "waits on guard/check open") || !strings.Contains(report, "next: work the rows it waits on first") {
		t.Fatalf("held row diagnosis:\n%s", report)
	}
	output.Reset()
	if err := diagnoseRow(root, "fresh/start", &output); err != nil {
		t.Fatal(err)
	}
	if report := output.String(); !strings.Contains(report, "state=open-ready") || !strings.Contains(report, "attempts: 0") || !strings.Contains(report, "next: dispatch it") {
		t.Fatalf("fresh row diagnosis:\n%s", report)
	}
	output.Reset()
	if err := diagnoseRow(root, "missing/row", &output); err != nil {
		t.Fatal(err)
	}
	if report := output.String(); !strings.Contains(report, "state=absent") {
		t.Fatalf("absent row diagnosis:\n%s", report)
	}
	if err := diagnoseRow(root, "malformed", &output); err == nil {
		t.Fatal("a reference without a step was accepted")
	}
}
