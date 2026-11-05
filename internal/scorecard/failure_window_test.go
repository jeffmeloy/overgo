package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/servingattempt"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestFailureRateCountsTheAttemptsOfItsWindow reads attempt histories, newest first, into the failed share. A recipe
// failing on and off under one commit reads as failing whatever its newest attempt did, where counting stopped at the
// newest success and read none. A fix closes its row with its first success, since failures of older code leave the
// count, and a failure after the fix is counted. Attempts past the window say nothing, and records that name no commit
// keep the rule they were written under: a success ends their failures.
func TestFailureRateCountsTheAttemptsOfItsWindow(t *testing.T) {
	t.Parallel()
	const old, fixed = "1111", "2222"
	ok := func(commit string) runrecord.ServingObservation {
		return runrecord.ServingObservation{Outcome: runrecord.OutcomeSucceeded, CodeCommit: commit}
	}
	bad := func(commit string) runrecord.ServingObservation {
		return runrecord.ServingObservation{Outcome: runrecord.OutcomeFailed, CodeCommit: commit}
	}
	cancelled := runrecord.ServingObservation{Outcome: runrecord.OutcomeCancelled, CodeCommit: old}
	repeat := func(observation runrecord.ServingObservation, count int) []runrecord.ServingObservation {
		history := make([]runrecord.ServingObservation, count)
		for index := range history {
			history[index] = observation
		}
		return history
	}
	then := func(histories ...[]runrecord.ServingObservation) []runrecord.ServingObservation {
		var joined []runrecord.ServingObservation
		for _, history := range histories {
			joined = append(joined, history...)
		}
		return joined
	}
	for _, reads := range []struct {
		name             string
		history          []runrecord.ServingObservation
		failed, attempts int
	}{
		{"broken 8 times of 9 under one commit, newest worked", then(repeat(ok(old), 1), repeat(bad(old), 8)), 8, 9},
		{"every failure older than the fix", then(repeat(ok(fixed), 1), repeat(bad(old), 8)), 0, 1},
		{"a failure since the fix", then(repeat(bad(fixed), 1), repeat(ok(fixed), 1), repeat(bad(old), 3)), 1, 2},
		{"failing and never fixed", repeat(bad(old), 3), 3, 3},
		{"failures past the window", then(repeat(ok(old), failureWindow), repeat(bad(old), 5)), 0, failureWindow},
		{"cancelled attempts", then(repeat(cancelled, 2), repeat(ok(old), 2)), 0, 2},
		{"records naming no commit", then(repeat(ok(""), 2), repeat(bad(""), 3)), 0, 2},
		{"a failure naming no commit after a success naming one", then(repeat(ok(fixed), 1), repeat(bad(""), 3)), 0, 1},
	} {
		summary := &servingSummary{}
		for _, attempt := range reads.history {
			summary.tally(attempt)
		}
		if summary.failures != reads.failed || summary.attempts != reads.attempts {
			t.Errorf("%s: read %d failed of %d, want %d of %d", reads.name, summary.failures, summary.attempts, reads.failed, reads.attempts)
		}
	}

	// The row a failing recipe opens closes with the first success of the fix, by the check that closes any row.
	record := func(history []runrecord.ServingObservation) Record {
		summary := &servingSummary{latest: testutil.ArtifactID(t, artifact.KindEvidence, "attempt")}
		for _, attempt := range history {
			summary.tally(attempt)
		}
		return Record{Rows: []Row{{
			Name: "recurring", Task: recipe.TaskInference, Cells: map[metric.Name]Cell{metric.Failures: failureCell(summary)},
		}}}
	}
	failing := func(gaps []Gap) bool {
		for _, gap := range gaps {
			if strings.Contains(gap.Detail, "recurring "+string(metric.Failures)) && strings.Contains(gap.Detail, "8 failed and 0 cancelled of the newest 9 attempts") {
				return true
			}
		}
		return false
	}
	if !failing(record(then(repeat(ok(old), 1), repeat(bad(old), 8))).Gaps(nil)) {
		t.Fatal("a recipe failing on and off under one commit opens no row")
	}
	if failing(record(then(repeat(ok(fixed), 1), repeat(bad(old), 8))).Gaps(nil)) {
		t.Fatal("the first success of a fix leaves its row open")
	}

	// An attempt says what code it ran, which all of the above reads.
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "attempted model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "attempted recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "attempted environment")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "attempt/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: recipeID}, {ID: environment}}}); err != nil {
		t.Fatal(err)
	}
	published, err := servingattempt.Record(ctx, store, runrecord.ServingObservation{
		Model: model, Recipe: recipeID, Environment: environment, Task: recipe.TaskInference,
		Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: 1,
	})
	if err != nil || published.CodeCommit == "" || published.CodeCommit != runrecord.RunningCommit() {
		t.Fatalf("a recorded attempt names commit %q (%v), want the one this process runs, %q", published.CodeCommit, err, runrecord.RunningCommit())
	}
}

// TestAVideoIsReadAtTheSizeAPersonAsksFor: an attempt at a person's size is an attempt, and stays out of the small request's
// reading; its time and peak memory read as cells of their own, compared on their own contract and printed by their column.
func TestAVideoIsReadAtTheSizeAPersonAsksFor(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "video model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "video recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "video environment")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "video/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: recipeID}, {ID: environment}}}); err != nil {
		t.Fatal(err)
	}
	const small, sized, smallPeak, sizedPeak = 13_000_000_000, 360_000_000_000, 3 << 30, 12 << 30
	attempt := func(label string, started int64, took uint64, peak uint64, userSize bool) {
		t.Helper()
		observation, err := runrecord.NewServingObservation(runrecord.ServingObservation{
			Model: model, Recipe: recipeID, Environment: environment, Task: recipe.TaskVideoGen, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: started, MeasuredNS: took, Resources: runrecord.ServingResources{PeakDeviceBytes: peak}, UserSize: userSize,
			Phases: []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: 1}},
		})
		testcheck.NoError(t, err)
		batch, err := observation.Batch("video/" + label)
		testcheck.NoError(t, err)
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	attempt("small", 1, small, smallPeak, false)
	attempt("sized", 2, sized, sizedPeak, true)
	summaries, err := servingByRecipe(ctx, store)
	testcheck.NoError(t, err)
	summary := summaries[recipeID]
	if summary == nil || summary.attempts != 2 || summary.peakDeviceBytes != smallPeak {
		t.Fatalf("the summary reads %+v, want both attempts and the small request's memory", summary)
	}
	if timed, measured := summary.timed(timingWholeOutput, "ms", "whole", "whole"); !measured || timed.Value != small/1e6 {
		t.Fatalf("the small request reads %+v (%t), want its own time", timed, measured)
	}
	unmeasured := gap("not read in this fixture")
	row := Row{Cells: map[metric.Name]Cell{metric.FirstOutput: unmeasured, metric.PeakMemory: unmeasured}, Warm: userSizeCells(summary)}
	timeCells, memoryCells := row.Measured(metric.FirstOutput), row.Measured(metric.PeakMemory)
	if len(timeCells) != 1 || timeCells[0].Value != sized/1e6 || !timeCells[0].AtUserSize() {
		t.Fatalf("the first-output column holds %+v, want the person's request", timeCells)
	}
	if len(memoryCells) != 1 || memoryCells[0].Value != sizedPeak || !memoryCells[0].AtUserSize() {
		t.Fatalf("the memory column holds %+v, want the person's request", memoryCells)
	}
}
