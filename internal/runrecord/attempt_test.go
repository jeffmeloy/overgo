package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func fixtureAttempt(t *testing.T) AttemptRecord {
	t.Helper()
	return AttemptRecord{
		PlanItem: "attempt-records", PlanStep: "do",
		Result:     testutil.ArtifactID(t, artifact.KindEvidence, "gate-result"),
		Recipe:     testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe"),
		CodeCommit: strings.Repeat("ab", 20),
		Outcome:    OutcomeSucceeded, WallNS: 42,
		Selection: AttemptSelection{Defined: 17, Selected: 12, Excluded: 5, CacheEligible: 3, CacheHits: 2, PlanningNS: 7},
		Diff:      AttemptDiff{Files: 2, Insertions: 40, Deletions: 3},
	}
}

// TestAttemptRecordRoundTrip pins the attempt contract: a validated
// record re-parses to the identical identity and observations.
func TestAttemptRecordRoundTrip(t *testing.T) {
	record, err := NewAttemptRecord(fixtureAttempt(t))
	if err != nil {
		t.Fatal(err)
	}
	content, err := record.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := attemptCodec.Parse(content.Data)
	if err != nil || parsed.ID != record.ID || parsed.PlanItem != "attempt-records" ||
		parsed.Selection.Excluded != 5 || parsed.Diff.Insertions != 40 {
		t.Fatalf("parsed attempt = (%+v, %v)", parsed, err)
	}
	if lineage := record.Lineage(); len(lineage) != 2 || lineage[0].Parent != record.Result {
		t.Fatalf("lineage = %+v", lineage)
	}
}

// TestAttemptRecordRefusesNonObservations pins fail-closed validation:
// an attempt cannot claim what the gate did not observe.
func TestAttemptRecordRefusesNonObservations(t *testing.T) {
	corrupt := map[string]func(*AttemptRecord){
		"missing plan step":    func(a *AttemptRecord) { a.PlanStep = "" },
		"slash in plan item":   func(a *AttemptRecord) { a.PlanItem = "x/y" },
		"result not evidence":  func(a *AttemptRecord) { a.Result = a.Recipe },
		"placeholder commit":   func(a *AttemptRecord) { a.CodeCommit = "HEAD-at-activation" },
		"success with failure": func(a *AttemptRecord) { a.Failure = "test" },
		"failure without step": func(a *AttemptRecord) { a.Outcome = OutcomeFailed },
		"zero wall":            func(a *AttemptRecord) { a.WallNS = 0 },
		"hits above eligible":  func(a *AttemptRecord) { a.Selection.CacheHits = 9 },
		"negative diff":        func(a *AttemptRecord) { a.Diff.Deletions = -1 },
		"invalid outcome":      func(a *AttemptRecord) { a.Outcome = "maybe" },
		"free-text strategy":   func(a *AttemptRecord) { a.Strategy = "two words" },
		"strategy not profile": func(a *AttemptRecord) { a.StrategyID = testutil.ArtifactID(t, artifact.KindEvidence, "strategy") },
		"task not recipe":      func(a *AttemptRecord) { a.TaskContract = testutil.ArtifactID(t, artifact.KindProfile, "task") },
		"environment not evidence": func(a *AttemptRecord) {
			a.Environment = testutil.ArtifactID(t, artifact.KindProfile, "environment")
		},
		"trajectory not evidence": func(a *AttemptRecord) { a.Trajectory = testutil.ArtifactID(t, artifact.KindRecipe, "trajectory") },
		"negative planning cost":  func(a *AttemptRecord) { a.Selection.PlanningNS = -1 },
	}
	for name, mutate := range corrupt {
		record := fixtureAttempt(t)
		mutate(&record)
		if _, err := NewAttemptRecord(record); err == nil {
			t.Fatalf("%s: accepted a non-observation", name)
		}
	}
}
