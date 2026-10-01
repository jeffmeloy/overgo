package main

import (
	"bytes"
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestScorecardRowsEveryServableModel holds the scorecard to its contract on
// a fixture store: every activation is a row, a cell the store's records
// measure shows its value, and every cell no record measures renders as a
// dash and is listed with its reason.
func TestScorecardRowsEveryServableModel(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	directory := scorecardFixture(t)
	var output bytes.Buffer
	if err := runArgs(ctx, []string{"-scorecard", "-repo", directory}, &output); err != nil {
		t.Fatal(err)
	}
	text := output.String()
	rows := map[string][]string{}
	for line := range strings.Lines(text) {
		cells := strings.Split(strings.Trim(strings.TrimSpace(line), "|"), "|")
		for index := range cells {
			cells[index] = strings.TrimSpace(cells[index])
		}
		if len(cells) == 2+len(scorecardHeadings) && (cells[1] == string(recipe.TaskForecast) || cells[1] == string(recipe.TaskImageGen)) {
			rows[cells[1]] = cells
		}
	}
	forecastRow, imageRow := rows[string(recipe.TaskForecast)], rows[string(recipe.TaskImageGen)]
	if forecastRow == nil || imageRow == nil {
		t.Fatalf("scorecard lacks an activation row:\n%s", text)
	}
	// Columns: model, task, first output, throughput, quality, failures,
	// load, peak memory, setup steps.
	if got := forecastRow[2]; got != "25 ms" {
		t.Errorf("first output = %q, want the median of the four completed samples, 25 ms", got)
	}
	if got := forecastRow[5]; got != "20% (1 of 5 attempts failed, 1 cancelled)" {
		t.Errorf("failures = %q, want the active recipe's attempts only", got)
	}
	for _, row := range [][]string{forecastRow, imageRow} {
		if got := row[8]; got != "3" {
			t.Errorf("%s setup steps = %q, want candidate, validated, active", row[1], got)
		}
	}
	if dashes := strings.Count(strings.Join(imageRow[2:], "|"), "—"); dashes != len(scorecardHeadings)-3 || imageRow[3] != "n/a" || imageRow[4] != "n/a" {
		t.Errorf("an image model never served shows %d dashes, want every cell but setup steps, its throughput, which does not apply, and its held-out quality, whose evaluation is only proposed: %v", dashes, imageRow)
	}
	for _, row := range [][]string{forecastRow, imageRow} {
		for index, metric := range []string{"First output", "Throughput", "Held-out quality", "Failures", "Load", "Peak memory", "Setup steps"} {
			if row[2+index] != "—" {
				continue
			}
			if !strings.Contains(text, "- "+row[0]+" "+row[1]+", "+metric+": ") {
				t.Errorf("the dash under %s for %s %s has no listed reason:\n%s", metric, row[0], row[1], text)
			}
		}
	}
}

// TestScorecardHasNoUnexplainedCell holds every published scorecard cell to
// a measurement or a stated reason: the fixture store's scorecard publishes
// with each cell naming the record that measured it or why none does, and a
// record with a cell that does neither is refused on publish and on read,
// so the loop never judges or proposes work from it.
func TestScorecardHasNoUnexplainedCell(t *testing.T) {
	t.Parallel()
	directory := scorecardFixture(t)
	var output bytes.Buffer
	if err := runArgs(t.Context(), []string{"-scorecard", "-publish", "-repo", directory}, &output); err != nil {
		t.Fatal(err)
	}
	records := scorecard.Directory(directory)
	published, err := scorecard.Records(records)
	if err != nil || len(published) != 1 || len(published[0].Rows) == 0 {
		t.Fatalf("published records = %+v, %v; want the fixture's one scorecard", published, err)
	}
	for _, row := range published[0].Rows {
		for _, column := range metric.Columns {
			if cell := row.Cells[column]; !cell.Source.Valid() && cell.Reason == "" {
				t.Fatalf("%s %s, %s is published with neither a measuring record nor a reason: %+v", row.Name, row.Task, column, cell)
			}
		}
	}
	later := time.Now().Add(time.Hour)
	for name, unexplain := range map[string]func(scorecard.Row){
		"a missing cell":             func(row scorecard.Row) { delete(row.Cells, metric.Load) },
		"a value without its record": func(row scorecard.Row) { row.Cells[metric.Load] = scorecard.Cell{Value: 1, Unit: "ms"} },
	} {
		record := published[0]
		record.Rows = slices.Clone(record.Rows)
		record.Rows[0].Cells = maps.Clone(record.Rows[0].Cells)
		unexplain(record.Rows[0])
		if _, err := scorecard.Publish(records, later, record); err == nil {
			t.Fatalf("a record with %s published", name)
		}
		if err := jsonfile.Write(filepath.Join(records, "scorecard-unexplained.json"), record, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := scorecard.Records(records); err == nil {
			t.Fatalf("a record with %s was read as one to judge", name)
		}
	}
}

// TestProposedGapVerifierReadsTheStore holds a proposed row's verifier to
// the scorecard the store holds now: a gap the store still shows fails it
// with the rows that fall short, and a gap its records close passes.
func TestProposedGapVerifierReadsTheStore(t *testing.T) {
	t.Parallel()
	directory := scorecardFixture(t)
	var output bytes.Buffer
	err := runArgs(t.Context(), []string{"-scorecard", "-repo", directory, "-gap", "scorecard-unmeasured-image-gen"}, &output)
	if err == nil || !strings.Contains(err.Error(), "no benchmark time to first token") {
		t.Fatalf("an image model never served passes its first-output verifier: %v", err)
	}
	// A task's gap lists each unmeasured cell: the first output the serving
	// attempts measure leaves it, while the task's other cells keep it open.
	err = runArgs(t.Context(), []string{"-scorecard", "-repo", directory, "-gap", "scorecard-unmeasured-forecast"}, &output)
	if err == nil || strings.Contains(err.Error(), "first-output") {
		t.Fatalf("the forecast gap still lists the first output the serving attempts measure: %v", err)
	}
	if err := runArgs(t.Context(), []string{"-scorecard", "-repo", directory, "-gap", "scorecard-failing-forecast"}, &output); err == nil {
		t.Fatal("a failed forecast attempt passes the failing-attempts verifier")
	}
}

// scorecardFixture publishes a store, in a directory of its own, whose
// scorecard has a served forecast model and an image model nobody served.
// Served attempts give the forecast model its time to first output (the
// median of skewed prefill samples, not their mean or newest) and its
// failure rate; a cancelled attempt and an attempt under a retired recipe
// move neither.
func scorecardFixture(t *testing.T) string {
	t.Helper()
	ctx := t.Context()
	directory := filepath.Join(t.TempDir(), "store")
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	forecastModel := testutil.ArtifactID(t, artifact.KindModel, "scorecard-forecast")
	imageModel := testutil.ArtifactID(t, artifact.KindModel, "scorecard-image")
	retired := testutil.ArtifactID(t, artifact.KindRecipe, "scorecard-retired-recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "scorecard-environment")
	for _, id := range []artifact.ID{forecastModel, imageModel, retired, environment} {
		testutil.PublishArtifact(t, store, id)
	}
	forecast, err := modelrecipe.CapabilityDefinition(recipe.TaskForecast, forecastModel)
	if err != nil {
		t.Fatal(err)
	}
	image, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorImagePrepare, imageModel, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	for key, definition := range map[string]recipe.Definition{"fixture/forecast": forecast, "fixture/image": image} {
		if err := modelrecipetest.PublishActivation(ctx, store, key, definition); err != nil {
			t.Fatal(err)
		}
	}
	for index, attempt := range []struct {
		recipe  artifact.ID
		outcome runrecord.Outcome
		failure string
		prefill time.Duration
	}{
		{forecast.ID, runrecord.OutcomeSucceeded, "", 10 * time.Millisecond},
		{forecast.ID, runrecord.OutcomeSucceeded, "", 900 * time.Millisecond},
		{forecast.ID, runrecord.OutcomeSucceeded, "", 30 * time.Millisecond},
		{forecast.ID, runrecord.OutcomeSucceeded, "", 20 * time.Millisecond},
		{forecast.ID, runrecord.OutcomeFailed, "device-lost", 0},
		{forecast.ID, runrecord.OutcomeCancelled, "", 5 * time.Second},
		{retired, runrecord.OutcomeFailed, "device-lost", 0},
		{retired, runrecord.OutcomeFailed, "device-lost", 0},
	} {
		observation := runrecord.ServingObservation{
			Version: artifact.InitialDocumentVersion, Model: forecastModel, Recipe: attempt.recipe, Environment: environment,
			Task: recipe.TaskForecast, Outcome: attempt.outcome, Failure: attempt.failure,
			StartedUnixNS: time.Unix(int64(index+1), 0).UnixNano(), MeasuredNS: uint64(10 * time.Second),
		}
		if attempt.prefill != 0 {
			observation.Hardware = []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: uint64(attempt.prefill)}}
		}
		if _, err := runrecord.PublishServingObservation(ctx, store, observation); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return directory
}
