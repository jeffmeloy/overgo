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

// TestAWarmRequestIsScoredBesideTheColdOne holds a row to both sides of a trade between a cold start and warm speed,
// as Qwen3.8 Flash Next's Strata port made one: its first request after load read 1742 ms to its first output and 66.8
// tok/s while its decode in a conversation rose to 136. A measurement's attempt that reused the session it had served
// is the warm request's alone: the row's first output and throughput stay its first request's, the newer warm attempt
// standing in for neither, and the row's warm cells are that attempt's prompt read through the cache and its decode's
// rate, each on its own contract. A warm cell the reference did not measure holds; a warm rate that falls is a decline
// of its own beside a cold throughput that holds.
func TestAWarmRequestIsScoredBesideTheColdOne(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "warm-model")
	served := testutil.ArtifactID(t, artifact.KindRecipe, "warm-recipe")
	here := testutil.ArtifactID(t, artifact.KindEvidence, "warm-environment")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "warm/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: served}, {ID: here}}}); err != nil {
		t.Fatal(err)
	}
	started := int64(0)
	// attempt commits a measurement's attempt, newer than every one before it: its first output, and its tokens over
	// its decode's milliseconds; a cold one timed its load, a warm one reused its session.
	attempt := func(warm bool, firstMS float64, tokens uint64, decodeMS float64) {
		t.Helper()
		started++
		ns := func(ms float64) uint64 { return uint64(math.Round(ms * float64(time.Millisecond))) }
		value := runrecord.ServingObservation{
			Model: model, Recipe: served, Environment: here, Task: recipe.TaskInference, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: started, MeasuredNS: ns(firstMS + decodeMS), SessionReused: warm, Usage: runrecord.ServingUsage{OutputTokens: tokens},
			Phases:   []runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: ns(decodeMS)}},
			Hardware: []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: ns(firstMS)}},
		}
		if !warm {
			value.Phases = append(value.Phases, runrecord.PhaseMetric{Phase: runrecord.PhaseLoad, DurationNS: ns(30000)})
		}
		observation, err := runrecord.NewServingObservation(value)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("warm/" + observation.ID.String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	attempt(false, 1742, 100, 1496)
	attempt(true, 50, 128, 941)
	summaries, err := servingByRecipe(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	summary := summaries[served]
	first := firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summary)
	rate := throughputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summary)
	if first.Value != 1742 || math.Abs(rate.Value-100/1.496) > 1e-6 {
		t.Fatalf("the row's first output and throughput = %v ms and %v tok/s, want its first request's, 1742 ms and %.4g", first.Value, rate.Value, 100/1.496)
	}
	warm := warmCells(recipe.TaskInference, summary)
	if len(warm) != 2 || warm[0].Value != 50 || warm[0].Contract.Metric != warmFirstMetric || math.Abs(warm[1].Value-128/0.941) > 1e-6 ||
		warm[1].Contract.Metric != warmRateMetric || warmColumn(warm[0]) != metric.FirstOutput || warmColumn(warm[1]) != metric.Throughput {
		t.Fatalf("the row's warm cells = %+v, want the warm attempt's 50 ms first output and %.4g tok/s, each on its own contract", warm, 128/0.941)
	}

	row := func(warm []Cell) Row {
		built := measuredFixture(model, map[metric.Name]Cell{metric.FirstOutput: first, metric.Throughput: rate})
		built.Warm = warm
		return built
	}
	if compared := (Pair{Before: new(row(nil)), After: new(row(warm))}).Compare(metric.Throughput); len(compared) != 0 {
		t.Errorf("warm cells beside a reference without them compare as %+v, want them to hold", compared)
	}
	slower := append([]Cell(nil), warm...)
	slower[1].Value = 90
	compared := (Pair{Before: new(row(warm)), After: new(row(slower))}).Compare(metric.Throughput)
	if len(compared) != 1 || compared[0].Verdict != regressed || compared[0].Then.Contract.Metric != warmRateMetric {
		t.Errorf("a warm rate of 90 after %.4g tok/s, its cold throughput held, compares as %+v, want the warm rate's decline alone", warm[1].Value, compared)
	}
}
