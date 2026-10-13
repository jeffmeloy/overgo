package scorecard

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestAContendedAttemptGivesTheScorecardNoReading holds what the scorecard reads once a measurement
// that saw other work on its machine has set its attempts aside.
//
// A recipe has three quiet attempts of 100 ms. A measurement beside other work then writes two
// slow successes and a failure. Read as they stand, the two slow attempts are most of the newest
// three and the cell reads the slower of them. Set aside from the store's sequence before that
// measurement, both successes give no reading: the cell reads 100 ms again, from the quiet
// attempts, and names one of them. The failure is not set aside and still counts.
//
// Setting the same attempts aside again changes nothing, a sequence nothing came after records
// nothing, and a record that names no competing work is refused.
func TestAContendedAttemptGivesTheScorecardNoReading(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "contended-model")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "contended-environment")
	served := testutil.ArtifactID(t, artifact.KindRecipe, "contended-recipe")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "contended/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: environment}, {ID: served}}}); err != nil {
		t.Fatal(err)
	}
	const millisecond = uint64(time.Millisecond)
	started := int64(0)
	// attempt commits one serving attempt of the recipe, newer than every one before it: a success
	// that timed its load and took so many milliseconds to its first output, or, given none, a failure.
	attempt := func(prefill uint64) artifact.ID {
		t.Helper()
		started++
		observation := runrecord.ServingObservation{
			Model: model, Recipe: served, Environment: environment, Task: recipe.TaskInference,
			Outcome: runrecord.OutcomeFailed, Failure: "device-lost", StartedUnixNS: started, MeasuredNS: millisecond,
		}
		if prefill != 0 {
			observation.Outcome, observation.Failure, observation.MeasuredNS = runrecord.OutcomeSucceeded, "", prefill*millisecond
			observation.Hardware = []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: prefill * millisecond}}
			observation.Phases = []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: millisecond}}
		}
		observation, err := runrecord.NewServingObservation(observation)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("contended/" + observation.ID.String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return observation.ID
	}
	// read is the recipe's first-output cell and its count of failures, as the scorecard reads them now.
	read := func() (Cell, int) {
		t.Helper()
		summaries, err := servingByRecipe(ctx, store)
		if err != nil || summaries[served] == nil {
			t.Fatalf("the recipe's attempts read as %v (%v)", summaries[served], err)
		}
		return firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summaries[served]), summaries[served].failures
	}

	quiet := []artifact.ID{attempt(100), attempt(100), attempt(100)}
	_, before := store.Head()
	attempt(900)
	slower := attempt(950)
	attempt(0)
	if cell, failures := read(); cell.Value != 900 || failures != 1 {
		t.Fatalf("before anything is set aside the cell reads %v ms with %d failures, want 900, the median of 950, 900 and 100, and the one failure", cell.Value, failures)
	}

	competing := []string{"compile.exe", "go.exe"}
	if recorded, err := SetAsideContended(ctx, store, before, nil); err == nil || !strings.Contains(err.Error(), "invalid contended reading") {
		t.Fatalf("a record that names no competing work answered (%d, %v), want it refused", recorded, err)
	}
	if recorded, err := SetAsideContended(ctx, store, before, competing); err != nil || recorded != 2 {
		t.Fatalf("setting aside what came after the quiet attempts recorded %d (%v), want the two successes and not the failure", recorded, err)
	}
	cell, failures := read()
	if cell.Value != 100 || failures != 1 {
		t.Fatalf("with the contended attempts set aside the cell reads %v ms with %d failures, want 100 from the quiet attempts and the failure still counted", cell.Value, failures)
	}
	if named := cell.Source; named == slower || named != quiet[0] && named != quiet[1] && named != quiet[2] {
		t.Fatalf("the cell names the attempt %s, want one of the quiet attempts %v", named, quiet)
	}
	if aside, err := setAside(ctx, store, slower); err != nil || !aside {
		t.Fatalf("the slower contended attempt reads as set aside: %v (%v), want true", aside, err)
	}
	if aside, err := setAside(ctx, store, quiet[0]); err != nil || aside {
		t.Fatalf("a quiet attempt reads as set aside: %v (%v), want false", aside, err)
	}

	if recorded, err := SetAsideContended(ctx, store, before, competing); err != nil || recorded != 2 {
		t.Fatalf("setting the same attempts aside again recorded %d (%v), want the same two and no failure", recorded, err)
	}
	if again, _ := read(); again.Value != cell.Value || again.Source != cell.Source {
		t.Fatalf("after the second setting aside the cell reads %v ms from %s, want it unchanged", again.Value, again.Source)
	}
	_, now := store.Head()
	if recorded, err := SetAsideContended(ctx, store, now, competing); err != nil || recorded != 0 {
		t.Fatalf("a sequence nothing came after recorded %d (%v), want none", recorded, err)
	}
}
