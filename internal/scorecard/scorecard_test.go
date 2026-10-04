package scorecard

import (
	"encoding/json"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestServingCountsWhatUsersHit holds the serving summary to the attempts
// users made: a failed attempt counts against its recipe until a later one
// succeeds, and one a gate result depends on -- a validation run's
// deliberate interrupt -- never does. The cells read the summary as a user
// meets the recipe now: the failure since its newest success in the two
// attempts back to it, the failure that success followed counted no longer,
// a whole-output capability's time to its output, and the device memory
// the success peaked at. Each keeps its origin and scope, the serving
// attempts of its recipe, so it compares only with the same population.
func TestServingCountsWhatUsersHit(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "served-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "served-recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "served-environment")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "serving/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: recipeID}, {ID: environment}}}); err != nil {
		t.Fatal(err)
	}
	observe := func(label string, attempt runrecord.ServingObservation) runrecord.ServingObservation {
		t.Helper()
		attempt.Model, attempt.Recipe, attempt.Environment, attempt.Task = model, recipeID, environment, recipe.TaskImageGen
		observation, err := runrecord.NewServingObservation(attempt)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("serving/" + label)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return observation
	}
	const peak, wholeOutputNS = 3 << 20, 2_000_000
	observe("fixed", runrecord.ServingObservation{Outcome: runrecord.OutcomeFailed, Failure: "execution_failed", StartedUnixNS: 1, MeasuredNS: 1})
	observe("success", runrecord.ServingObservation{Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 2, MeasuredNS: wholeOutputNS,
		Resources: runrecord.ServingResources{PeakDeviceBytes: peak}})
	interrupt := observe("interrupt", runrecord.ServingObservation{Outcome: runrecord.OutcomeFailed, Failure: "execution_failed", StartedUnixNS: 3, MeasuredNS: 1})
	observe("user", runrecord.ServingObservation{Outcome: runrecord.OutcomeFailed, Failure: "execution_failed", StartedUnixNS: 4, MeasuredNS: 1})
	gate, err := runrecord.NewGateRecord(recipeID, environment, "0123456789abcdef0123456789abcdef01234567", runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{{Name: "interrupts", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := gate.Batch("validation/interrupts")
	if err != nil {
		t.Fatal(err)
	}
	batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(gate.Result.ID, interrupt.ID)...)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	summaries, err := servingByRecipe(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	summary := summaries[recipeID]
	if summary == nil || summary.attempts != 2 || summary.failures != 1 {
		t.Fatalf("serving summary = %+v, want the user's two attempts back to the newest success, not the validation run's nor the failure that success followed", summary)
	}
	var none evaluation.BenchmarkSummary
	if cell := failureCell(summary); cell.Value != 0.5 || cell.Label != "1 failed and 0 cancelled since the last success" || cell.Contract.Origin != originServing || cell.Contract.Scope != recipeID {
		t.Fatalf("failure cell = %+v, want one failure in two attempts, scoped to the recipe's serving attempts", cell)
	}
	if cell := failureCell(&servingSummary{attempts: 2, failures: 2}); cell.Value != 1 || cell.Label != "2 failed and 0 cancelled and none succeeded" {
		t.Fatalf("failure cell of a recipe that never succeeded = %+v, want every attempt failed", cell)
	}
	if cell := firstOutputCell(recipe.TaskImageGen, none, false, summary); cell.Value != milliseconds(wholeOutputNS) || cell.Unit != "ms" {
		t.Fatalf("first-output cell = %+v, want the whole output's time", cell)
	}
	if cell := peakMemoryCell(none, false, summary); cell.Value != peak {
		t.Fatalf("peak-memory cell = %+v, want %d bytes", cell, peak)
	}
}

// TestServingCellsReadTheNewestMeasuredAttempt holds the timing and memory
// cells to one attempt: the newest success that timed its model's load, as
// the cycle measures a recipe. An older, slower measured attempt is not
// blended in, so a landed improvement reads in full; a later request served
// from a resident session, which times no load, does not stand in for the
// measurement; and a stale benchmark record is read only where no attempt
// measured the cell. A recipe whose attempts never timed a load reads its
// newest success.
func TestServingCellsReadTheNewestMeasuredAttempt(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "measured-model")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "measured-environment")
	measuredRecipe, served := testutil.ArtifactID(t, artifact.KindRecipe, "measured-recipe"), testutil.ArtifactID(t, artifact.KindRecipe, "served-recipe")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "measured/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: environment}, {ID: measuredRecipe}, {ID: served}}}); err != nil {
		t.Fatal(err)
	}
	const millisecond, peak = uint64(time.Millisecond), 5 << 30
	attempt := func(label string, activation artifact.ID, started int64, loadMS, prefillMS, decodeMS, tokens, devicePeak uint64) artifact.ID {
		t.Helper()
		value := runrecord.ServingObservation{
			Model: model, Recipe: activation, Environment: environment, Task: recipe.TaskInference,
			Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: started, MeasuredNS: (prefillMS + decodeMS) * millisecond,
			Usage:     runrecord.ServingUsage{OutputTokens: tokens},
			Resources: runrecord.ServingResources{PeakDeviceBytes: devicePeak},
			Hardware:  []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: prefillMS * millisecond}},
			Phases:    []runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: decodeMS * millisecond}},
		}
		if loadMS != 0 {
			value.Phases = append(value.Phases, runrecord.PhaseMetric{Phase: runrecord.PhaseLoad, DurationNS: loadMS * millisecond})
		}
		observation, err := runrecord.NewServingObservation(value)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("measured/" + label)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return observation.ID
	}
	attempt("slow", measuredRecipe, 1, 9000, 200, 2000, 10, 2*peak)
	fast := attempt("fast", measuredRecipe, 2, 4000, 100, 1000, 80, peak)
	latest := attempt("page", measuredRecipe, 3, 0, 5, 200, 50, 0)
	attempt("page-older", served, 1, 0, 7, 400, 40, 0)
	page := attempt("page-newest", served, 2, 0, 5, 200, 50, 0)

	summaries, err := servingByRecipe(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	stale := evaluation.BenchmarkSummary{FirstTokenMS: 250, DecodeTokensPerSecond: 4, LoadMS: 12000, PeakDeviceBytes: 2 * peak}
	comparison := testutil.ArtifactID(t, artifact.KindProfile, "benchmark-execution")
	staleData, err := json.Marshal(map[string]any{"record": latest, "code_commit": "0123456789abcdef0123456789abcdef01234567", "environment": environment, "comparison": comparison})
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(staleData, &stale); err != nil {
		t.Fatal(err)
	}
	summary := summaries[measuredRecipe]
	for name, cell := range map[string]Cell{
		"first output": firstOutputCell(recipe.TaskInference, stale, true, summary), "throughput": throughputCell(recipe.TaskInference, stale, true, summary),
		"load": loadCell(stale, true, summary), "peak": peakMemoryCell(stale, true, summary),
	} {
		want := map[string]float64{"first output": 100, "throughput": 80, "load": 4000, "peak": peak}[name]
		if cell.Value != want || cell.Source != fast || cell.Contract.Origin != originServing {
			t.Errorf("%s = %+v, want %v from the newest attempt that timed its load", name, cell, want)
		}
	}
	if cell := failureCell(summary); cell.Value != 0 || cell.Source != latest {
		t.Errorf("failure cell = %+v, want none failed since the newest success, read from the latest attempt", cell)
	}
	if cell := firstOutputCell(recipe.TaskInference, stale, true, summaries[served]); cell.Value != 5 || cell.Source != page {
		t.Errorf("first output of a recipe no attempt loaded = %+v, want its newest success", cell)
	}
	if cell := loadCell(stale, true, summaries[served]); cell.Value != stale.LoadMS || cell.Contract.Origin != originBenchmark {
		t.Errorf("load of a recipe no attempt loaded = %+v, want the benchmark's", cell)
	}
	for name, cell := range map[string]Cell{
		"first output": firstOutputCell(recipe.TaskInference, stale, true, nil), "throughput": throughputCell(recipe.TaskInference, stale, true, nil),
		"load": loadCell(stale, true, nil), "peak": peakMemoryCell(stale, true, nil),
	} {
		if cell.Source != latest || cell.CodeCommit != "0123456789abcdef0123456789abcdef01234567" || cell.Environment != environment || cell.Contract.Comparison != comparison {
			t.Errorf("benchmark %s lost its producing authorities: %+v", name, cell)
		}
	}
}

// TestPeakMemoryReadsASampledZero holds the peak-memory cell to what the
// attempts of a recipe measured. An attempt that sampled its device at its
// finish and held nothing measured zero: its model runs on the host, and
// the cell says so on the serving contract of its recipe. An attempt that
// sampled nothing, or whose only sample is the prefill that times its first
// output, measured no peak and stays a gap. Samples that held memory read
// as their peak, whatever the task, and a benchmark's peak fills in only
// where no attempt measured one.
func TestPeakMemoryReadsASampledZero(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const wall, resident, peak = 2_000_000, 1 << 20, 3 << 20
	attempts := map[string][]runrecord.ServingHardwareSample{
		"host":      {{Stage: runrecord.ServingHardwareFinish, ElapsedNS: wall}},
		"unsampled": nil,
		"prefill":   {{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: wall / 2}},
		"device": {
			{Stage: runrecord.ServingHardwareStart, DeviceCurrentBytes: resident, DevicePeakBytes: resident, DeviceAllocations: 1},
			{Stage: runrecord.ServingHardwareFinish, ElapsedNS: wall, DeviceCurrentBytes: resident, DevicePeakBytes: peak, DeviceAllocations: 1},
		},
	}
	model := testutil.ArtifactID(t, artifact.KindModel, "sampled-model")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "sampled-environment")
	authorities := []artifact.Descriptor{{ID: model}, {ID: environment}}
	recipes, observed := map[string]artifact.ID{}, map[string]artifact.ID{}
	for name := range attempts {
		recipes[name] = testutil.ArtifactID(t, artifact.KindRecipe, "sampled-recipe-"+name)
		authorities = append(authorities, artifact.Descriptor{ID: recipes[name]})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "sampled/authorities", Artifacts: authorities}); err != nil {
		t.Fatal(err)
	}
	for name, hardware := range attempts {
		observation, err := runrecord.NewServingObservation(runrecord.ServingObservation{
			Model: model, Recipe: recipes[name], Environment: environment, Task: recipe.TaskForecast,
			Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: wall, Hardware: hardware,
		})
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("sampled/" + name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		observed[name] = observation.ID
	}
	summaries, err := servingByRecipe(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	var none evaluation.BenchmarkSummary
	host := summaries[recipes["host"]]
	want := Cell{Unit: "bytes", Label: "0 bytes: runs on the host", Source: observed["host"],
		Contract: Contract{Metric: "peak device memory", Origin: originServing, Scope: recipes["host"]}}
	if cell := peakMemoryCell(none, false, host); !cell.Measured() || cell != want {
		t.Fatalf("peak-memory cell of an attempt that sampled a device holding nothing = %+v, want the measured zero %+v", cell, want)
	}
	if cell := firstOutputCell(recipe.TaskForecast, none, false, host); cell.Value != milliseconds(wall) {
		t.Fatalf("first-output cell = %+v, want the whole output's time: the finish sample is no prefill", cell)
	}
	for _, name := range []string{"unsampled", "prefill"} {
		if cell := peakMemoryCell(none, false, summaries[recipes[name]]); cell.Measured() || cell.Reason != "no benchmark or serving peak device memory" {
			t.Fatalf("peak-memory cell of the %s attempt = %+v, want the gap: nothing sampled its device at the finish", name, cell)
		}
	}
	benchmarked := evaluation.BenchmarkSummary{PeakDeviceBytes: 2 * peak}
	for _, withBenchmark := range []bool{false, true} {
		if cell := peakMemoryCell(benchmarked, withBenchmark, summaries[recipes["device"]]); cell.Value != peak || cell.Contract.Origin != originServing {
			t.Fatalf("peak-memory cell of samples that held memory = %+v, want the attempt's %d-byte peak, a benchmark beside it or not", cell, peak)
		}
	}
	if cell := peakMemoryCell(benchmarked, true, host); cell != want {
		t.Fatalf("peak-memory cell of a measured zero beside a benchmark = %+v, want the attempt's zero", cell)
	}
	if cell := peakMemoryCell(benchmarked, true, summaries[recipes["unsampled"]]); cell.Value != 2*peak || cell.Contract.Origin != originBenchmark {
		t.Fatalf("peak-memory cell no attempt measured = %+v, want the benchmark's %d bytes", cell, 2*peak)
	}
}
