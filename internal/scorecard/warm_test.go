package scorecard

import (
	"math"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestSpeedIsReadWarmAtStatedLengths holds an inference row's speed to its warm readings: the owner, 2026-11-07, cold
// numbers are not reported. Qwen3.8 Flash Next's first request after load read 1742 ms to its first output and 66.8
// tok/s, while on the loaded model a request read again decodes at 136 tok/s and a fresh 4,096-token prompt, its size
// read once before, prefills in 1610 ms. The row's first output and throughput are the warm request's, each on the warm
// contract, and its 4,096-token cell is the measured prompt's, so its prompt processing at that size is 4,096 tokens
// over that time; the first request after load stands for neither, and a row whose model took no warm request keeps
// its first request's on that reading's own contract, which no report of warm speed reads. A warm rate that falls is
// the row's throughput decline.
func TestSpeedIsReadWarmAtStatedLengths(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	model := testutil.ArtifactID(t, artifact.KindModel, "warm-model")
	served := testutil.ArtifactID(t, artifact.KindRecipe, "warm-recipe")
	coldOnly := testutil.ArtifactID(t, artifact.KindRecipe, "cold-recipe")
	here := testutil.ArtifactID(t, artifact.KindEvidence, "warm-environment")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "warm/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: served}, {ID: coldOnly}, {ID: here}}}); err != nil {
		t.Fatal(err)
	}
	started := int64(0)
	// reading is one attempt of a measurement: the tokens it read, its first output, and its tokens over its decode's
	// milliseconds; the first request timed its load, and a warm one reused its session.
	type reading struct {
		served        artifact.ID
		loaded, warm  bool
		input, tokens uint64
		firstMS       float64
		decodeMS      float64
	}
	attempt := func(read reading) {
		t.Helper()
		started++
		ns := func(ms float64) uint64 { return uint64(math.Round(ms * float64(time.Millisecond))) }
		value := runrecord.ServingObservation{
			Model: model, Recipe: read.served, Environment: here, Task: recipe.TaskInference, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: started, MeasuredNS: ns(read.firstMS + read.decodeMS), SessionReused: read.warm,
			Usage:    runrecord.ServingUsage{InputTokens: read.input, OutputTokens: read.tokens},
			Phases:   []runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: ns(read.decodeMS)}},
			Hardware: []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: ns(read.firstMS)}},
		}
		if read.loaded {
			value.Phases = append(value.Phases, runrecord.PhaseMetric{Phase: runrecord.PhaseLoad, DurationNS: ns(30000)})
		}
		observation := testcheck.Must(runrecord.NewServingObservation(value))(t)
		batch := testcheck.Must(observation.Batch("warm/" + observation.ID.String()))(t)
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	prompt := evaluation.PromptLengths[0]
	attempt(reading{served: served, loaded: true, input: 1, firstMS: 1742, tokens: 100, decodeMS: 1496})
	attempt(reading{served: served, warm: true, input: uint64(prompt), firstMS: 1610, tokens: 24, decodeMS: 180})
	attempt(reading{served: served, warm: true, input: 256, firstMS: 50, tokens: 128, decodeMS: 941})
	attempt(reading{served: coldOnly, loaded: true, input: 1, firstMS: 1742, tokens: 100, decodeMS: 1496})
	summaries := testcheck.Must(servingByRecipe(ctx, store))(t)
	summary := summaries[served]
	first := firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summary)
	rate := throughputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, summary)
	if first.Value != 50 || first.Contract.Metric != warmFirstMetric || math.Abs(rate.Value-128/0.941) > 1e-6 || rate.Contract.Metric != warmRateMetric {
		t.Fatalf("the row's first output and throughput = %+v and %+v, want the warm request's 50 ms and %.4g tok/s on the warm contract", first, rate, 128/0.941)
	}
	lengths := lengthCells(recipe.TaskInference, summary)
	if len(lengths) == 0 || lengths[0].Prompt != prompt || lengths[0].Value != 1610 {
		t.Fatalf("the row's longer-prompt cells = %+v, want the %d-token prompt's 1610 ms", lengths, prompt)
	}
	cold := summaries[coldOnly]
	for column, cell := range map[metric.Name]Cell{
		metric.FirstOutput: firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, cold),
		metric.Throughput:  throughputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, cold),
	} {
		if cell.Contract.Metric == warmFirstMetric || cell.Contract.Metric == warmRateMetric {
			t.Errorf("a row with no warm request reports its %s on the warm contract: %+v", column, cell)
		}
	}

	row := func(rate Cell) Row {
		return measuredFixture(model, map[metric.Name]Cell{metric.FirstOutput: first, metric.Throughput: rate})
	}
	slower := rate
	slower.Value = 90
	compared := (Pair{Before: new(row(rate)), After: new(row(slower))}).Compare(metric.Throughput)
	if len(compared) != 1 || compared[0].Verdict != regressed || compared[0].Then.Contract.Metric != warmRateMetric {
		t.Errorf("a warm rate of 90 after %.4g tok/s compares as %+v, want the row's throughput decline", rate.Value, compared)
	}
}

// TestAWarmRateNeedsEnoughTokens holds a warm decode rate to a reading that generated at least rateTokens: Qwen2.5-0.5B's
// warm reading of one token in 85.1 microseconds read 11,751 tok/s, which is the decode's start and no rate, so its row
// reads no warm rate from it, and a newer warm reading of 128 tokens over 381.6 ms gives the rate, 335 tok/s.
func TestAWarmRateNeedsEnoughTokens(t *testing.T) {
	t.Parallel()
	warm := func(tokens uint64, decode time.Duration) runrecord.ServingObservation {
		return runrecord.ServingObservation{
			SessionReused: true, MeasuredNS: uint64(time.Millisecond + decode), Usage: runrecord.ServingUsage{InputTokens: 256, OutputTokens: tokens},
			Phases:   []runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: uint64(decode)}},
			Hardware: []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: uint64(time.Millisecond)}},
		}
	}
	first := runrecord.ServingObservation{MeasuredNS: uint64(time.Second), Usage: runrecord.ServingUsage{InputTokens: 5, OutputTokens: 4},
		Hardware: []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: uint64(time.Millisecond)}}}
	_, short := servedAttempts(t, []runrecord.ServingObservation{first, warm(1, 85100*time.Nanosecond)})
	if rate := throughputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, short); rate.Contract.Metric == warmRateMetric {
		t.Fatalf("a warm reading of one token gave the rate %+v", rate)
	}
	_, whole := servedAttempts(t, []runrecord.ServingObservation{first, warm(1, 85100*time.Nanosecond), warm(128, 381600*time.Microsecond)})
	if rate := throughputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, whole); rate.Contract.Metric != warmRateMetric || math.Abs(rate.Value-128/0.3816) > 1e-6 {
		t.Fatalf("a warm reading of 128 tokens over 381.6 ms gave %+v, want %.4g tok/s on the warm contract", rate, 128/0.3816)
	}
}
