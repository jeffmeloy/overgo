package scorecard

import (
	"math"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestOneBusyAttemptCannotMaskADecline holds a serving timing cell to the middle of its recipe's newest
// three attempts that timed their load, and its spread to how far those stand from that middle, so one
// attempt taken while the machine was busy does no harm either way. gemma-4-12B-it-fp8's published first
// output read 157.34 ms, then 308.4 ms, a disturbed reading, then 132.15 ms. Read as the newest attempt,
// with the range of three over their mean as its spread, the 308.4 filed a decline that was none and then
// stood in the next cell's spread of 0.88, under which a reading of 600 ms filed nothing. Read as the
// median, the busy attempt leaves the cell at 157.34 ms and files nothing, and 600 ms after it is a
// decline. A whole-output cell whose attempts read 340, 13 and 14 ms carries a spread of 2.67 that way, as
// one published cell did, and under it no reading at all could file; five times its 14 ms is a decline
// now. The 9 ms operation that read 8.96, 16.37 and 23.03 ms at three publications and 49.12, 18.07 and
// 30.51 ms at three repeats on an idle machine files nothing at any of them, judged one at a time, where
// the newest attempt alone filed at the second. An attempt taken under another environment is no repeat
// of the cell's: it joins neither the median nor the spread. Each cell names the attempt its value is the
// reading of: the middle one of three, the only one of one, and of two, whose median lies midway and is
// neither's reading, the newer, the one the row's memory is read from. What the median costs is held too:
// three attempts of 100 ms and then two of 160 are a decline at the second, one publication late.
func TestOneBusyAttemptCannotMaskADecline(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "busy-model")
	here, elsewhere := testutil.ArtifactID(t, artifact.KindEvidence, "busy-environment"), testutil.ArtifactID(t, artifact.KindEvidence, "other-environment")
	text, whole := testutil.ArtifactID(t, artifact.KindRecipe, "busy-text-recipe"), testutil.ArtifactID(t, artifact.KindRecipe, "busy-whole-recipe")
	noisy, moved := testutil.ArtifactID(t, artifact.KindRecipe, "busy-noisy-recipe"), testutil.ArtifactID(t, artifact.KindRecipe, "busy-moved-recipe")
	stepped := testutil.ArtifactID(t, artifact.KindRecipe, "busy-stepped-recipe")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "busy/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: here}, {ID: elsewhere}, {ID: text}, {ID: whole}, {ID: noisy}, {ID: moved}, {ID: stepped}}}); err != nil {
		t.Fatal(err)
	}
	started := int64(0)
	// attempt commits one attempt that timed its load, newer than every one before it. The text
	// recipe's samples its prefill at the reading; another recipe's returns its whole output then.
	attempt := func(served, environment artifact.ID, ms float64) artifact.ID {
		t.Helper()
		started++
		elapsed := uint64(math.Round(ms * float64(time.Millisecond)))
		value := runrecord.ServingObservation{
			Model: model, Recipe: served, Environment: environment, Task: recipe.TaskImageGen,
			Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: started, MeasuredNS: elapsed,
			Phases: []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: uint64(time.Millisecond)}},
		}
		if served == text {
			value.Task = recipe.TaskInference
			value.Hardware = []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: elapsed}}
		}
		observation, err := runrecord.NewServingObservation(value)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("busy/" + observation.ID.String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return observation.ID
	}
	// published is a recipe's first-output cell as a scorecard published now would carry it.
	published := func(served artifact.ID) Cell {
		t.Helper()
		summaries, err := servingByRecipe(ctx, store)
		if err != nil {
			t.Fatal(err)
		}
		return firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summaries[served])
	}
	// at is a later reading of a published cell: its contract and environment, another value.
	at := func(cell Cell, value float64) Cell {
		cell.Value, cell.Spread = value, 0
		return cell
	}
	near := func(got, want float64) bool { return math.Abs(got-want) <= 1e-9 }

	// The text cell rests on two quiet attempts, then the busy one arrives, then a quiet one again.
	attempt(text, here, 128)
	quiet := attempt(text, here, 157.34)
	two := published(text)
	if !near(two.Value, (128+157.34)/2) || !near(two.Spread, (157.34-128)/(157.34+128)) || two.Source != quiet {
		t.Errorf("first output of attempts of 128 and 157.34 ms = %v with spread %v from %s, want the %v midway between them, their half-gap over it, and the newer attempt %s named", two.Value, two.Spread, two.Source, (128+157.34)/2, quiet)
	}
	attempt(text, here, 308.4)
	busy := published(text)
	if busy.Value != 157.34 || busy.Source != quiet {
		t.Errorf("first output with a busy attempt of 308.4 ms newest = %v from %s, want the median 157.34 from the attempt that read it, %s", busy.Value, busy.Source, quiet)
	}
	attempt(text, here, 132.15)
	reference := published(text)
	if reference.Value != 157.34 || !near(reference.Spread, (157.34-132.15)/157.34) || reference.Source != quiet {
		t.Errorf("first output of attempts of 157.34, 308.4 and 132.15 ms = %v with spread %v from %s, want the median 157.34 with spread %v from %s: the busy attempt widens nothing", reference.Value, reference.Spread, reference.Source, (157.34-132.15)/157.34, quiet)
	}
	if verdict, why := holds(metric.FirstOutput, reference, at(reference, 600)); verdict != regressed {
		t.Errorf("600 ms after attempts of 157.34, 308.4 and 132.15 ms, the reference at %v with spread %v: holds = %q (%s), want %q: the busy attempt masks the decline", reference.Value, reference.Spread, verdict, why, regressed)
	}

	// One whole-output attempt some 24 times the other two.
	for _, ms := range []float64{340, 13, 14} {
		attempt(whole, here, ms)
	}
	slow := published(whole)
	if slow.Value != 14 || !near(slow.Spread, 1.0/14) {
		t.Errorf("whole output of attempts of 340, 13 and 14 ms = %v with spread %v, want the median 14 with spread 1/14", slow.Value, slow.Spread)
	}
	if verdict, why := holds(metric.FirstOutput, slow, at(slow, 70)); verdict != regressed {
		t.Errorf("70 ms after whole outputs of 340, 13 and 14 ms, the reference at %v with spread %v: holds = %q (%s), want %q: no reading could file under a spread above one", slow.Value, slow.Spread, verdict, why, regressed)
	}

	// The noisy operation, a publication an attempt. Its six publications are one cell's readings, and
	// what they allow the next is the reference's to say: 46 ms holds what they held, and 70 does not.
	var readings []Record
	for index, step := range []struct{ ms, median float64 }{{8.96, 8.96}, {16.37, (8.96 + 16.37) / 2}, {23.03, 16.37}, {49.12, 23.03}, {18.07, 23.03}, {30.51, 30.51}} {
		newest := attempt(noisy, here, step.ms)
		cell := published(noisy)
		if !near(cell.Value, step.median) {
			t.Errorf("the noisy operation after its reading of %v ms = %v, want the median %v", step.ms, cell.Value, step.median)
		}
		if index == 0 && (cell.Source != newest || cell.Spread != 0) {
			t.Errorf("a cell of one attempt = spread %v from %s, want no spread and that attempt, %s", cell.Spread, cell.Source, newest)
		}
		readings = append(readings, Record{Rows: []Row{measuredFixture(noisy, map[metric.Name]Cell{metric.FirstOutput: cell})}})
	}
	// The five distinct readings are 8.96, 12.665, 16.37, 23.03 and 30.51 ms: the level is 12.665, the
	// best three in a row held, and eight times their median step of 5.1825 ms is allowed beyond it.
	against := Reference(append(readings, Record{})).Rows[0].Cells[metric.FirstOutput]
	if !near(against.Value, (8.96+16.37)/2) || !near(against.allowed, noiseMultiple*(3.705+6.66)/2) {
		t.Errorf("the noisy operation's reference = %v allowing %v, want its best level 12.665 allowing eight times the 5.1825 its readings step by", against.Value, against.allowed)
	}
	for ms, want := range map[float64]Verdict{54: holding, 55: regressed} {
		if verdict, why := holds(metric.FirstOutput, against, at(against, ms)); verdict != want {
			t.Errorf("the noisy operation at %v ms: holds = %q (%s), want %q: its own readings allow 41.46 ms beyond 12.665", ms, verdict, why, want)
		}
	}

	// Two attempts under another environment lie between the two taken under this one.
	attempt(moved, here, 90)
	attempt(moved, elsewhere, 400)
	attempt(moved, elsewhere, 410)
	newest := attempt(moved, here, 100)
	if cell := published(moved); !near(cell.Value, 95) || !near(cell.Spread, 5.0/95) || cell.Source != newest || cell.Environment != here {
		t.Errorf("attempts of 90 ms here, 400 and 410 ms under another environment, then 100 ms here = %v with spread %v from %s under %s, want the 95 ms midway between the two taken here, spread 5/95, from the newer, %s", cell.Value, cell.Spread, cell.Source, cell.Environment, newest)
	}

	// A step that stays: the first slow attempt is one of three, the second moves the median.
	for _, ms := range []float64{100, 100, 100} {
		attempt(stepped, here, ms)
	}
	steady := published(stepped)
	attempt(stepped, here, 160)
	once := published(stepped)
	if verdict, why := holds(metric.FirstOutput, steady, once); once.Value != 100 || verdict != holding {
		t.Errorf("one attempt of 160 ms after three of 100 = %v: holds = %q (%s), want 100 and %q: one slow attempt is not yet a step, and the reading has not moved", once.Value, verdict, why, holding)
	}
	attempt(stepped, here, 160)
	twice := published(stepped)
	if verdict, why := holds(metric.FirstOutput, once, twice); twice.Value != 160 || verdict != regressed {
		t.Errorf("a second attempt of 160 ms after three of 100 = %v: holds = %q (%s), want 160 and %q: a step that stays is a decline, one publication late", twice.Value, verdict, why, regressed)
	}
}
