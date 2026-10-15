package scorecard

import (
	"slices"
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

// TestAContendedAttemptGivesTheScorecardNoReading preserves quiet evidence and
// failed-attempt accounting while excluding exact successful producer IDs.
// Unrelated observations arrive before admission, during work and after it.
// A measured observation declared bare before quiet evidence is filled later;
// its original descriptor order must not hide it. Replaying the exclusions is
// idempotent, and an empty identity list records nothing.
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
	other := testutil.ArtifactID(t, artifact.KindRecipe, "unrelated-recipe")
	producer := served
	if _, err := store.Commit(ctx, artifact.Batch{Key: "contended/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: environment}, {ID: served}, {ID: other}}}); err != nil {
		t.Fatal(err)
	}
	const millisecond = uint64(time.Millisecond)
	started := int64(0)
	var deferred artifact.Batch
	deferContent := false
	// attempt commits one serving attempt of the recipe, newer than every one before it: a success
	// that timed its load and took so many milliseconds to its first output, or, given none, a failure.
	attempt := func(prefill uint64) artifact.ID {
		t.Helper()
		started++
		observation := runrecord.ServingObservation{
			Model: model, Recipe: producer, Environment: environment, Task: recipe.TaskInference,
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
		if deferContent {
			deferred = batch
			batch = artifact.Batch{Key: "bare/" + observation.ID.String(), Artifacts: []artifact.Descriptor{{ID: observation.ID}}}
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

	deferContent = true
	late := attempt(975)
	deferContent = false
	quiet := []artifact.ID{attempt(100), attempt(100), attempt(100)}
	waiting := attempt(100)
	slow := attempt(900)
	concurrent := attempt(100)
	slower := attempt(950)
	failed := attempt(0)
	if cell, failures := read(); cell.Value != 900 || failures != 1 {
		t.Fatalf("before anything is set aside the cell reads %v ms with %d failures, want 900, the median of 950, 900 and 100, and the one failure", cell.Value, failures)
	}

	producer = other
	after := attempt(100)
	producer = served
	if _, err := store.Commit(ctx, deferred); err != nil {
		t.Fatal(err)
	}
	quiet = append(quiet, waiting, concurrent, after)
	attempts := []artifact.ID{slow, slower, late, failed}
	competing := []string{"compile.exe", "go.exe"}
	if recorded, err := SetAsideContended(ctx, store, attempts, nil); err == nil || !strings.Contains(err.Error(), "invalid contended reading") {
		t.Fatalf("a record that names no competing work answered (%d, %v), want it refused", recorded, err)
	}
	if recorded, err := SetAsideContended(ctx, store, attempts, competing); err != nil || recorded != 3 {
		t.Errorf("setting aside the exact measured attempts recorded %d (%v), want the three measured successes and not the failure", recorded, err)
	}
	cell, failures := read()
	if cell.Value != 100 || failures != 1 {
		t.Fatalf("with the contended attempts set aside the cell reads %v ms with %d failures, want 100 from the quiet attempts and the failure still counted", cell.Value, failures)
	}
	if named := cell.Source; named == slower || !slices.Contains(quiet, named) {
		t.Fatalf("the cell names the attempt %s, want one of the quiet attempts %v", named, quiet)
	}
	if aside, err := setAside(ctx, store, slower); err != nil || !aside {
		t.Fatalf("the slower contended attempt reads as set aside: %v (%v), want true", aside, err)
	}
	if aside, err := setAside(ctx, store, quiet[0]); err != nil || aside {
		t.Fatalf("a quiet attempt reads as set aside: %v (%v), want false", aside, err)
	}

	for _, id := range append(slices.Clone(quiet), attempts...) {
		aside, err := setAside(ctx, store, id)
		want := slices.Contains(attempts, id) && id != failed
		if err != nil || aside != want {
			t.Errorf("attempt %s set aside = %t (%v), want %t", id, aside, err, want)
		}
	}

	if recorded, err := SetAsideContended(ctx, store, attempts, competing); err != nil || recorded != 3 {
		t.Fatalf("setting the same attempts aside again recorded %d (%v), want the same three and no failure", recorded, err)
	}
	if again, _ := read(); again.Value != cell.Value || again.Source != cell.Source {
		t.Fatalf("after the second setting aside the cell reads %v ms from %s, want it unchanged", again.Value, again.Source)
	}
	if recorded, err := SetAsideContended(ctx, store, nil, competing); err != nil || recorded != 0 {
		t.Fatalf("an empty attempt list recorded %d (%v), want none", recorded, err)
	}
}
