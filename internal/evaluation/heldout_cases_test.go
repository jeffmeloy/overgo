package evaluation

import (
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
)

// TestHeldoutVerdictSummaryCountsItsRecords judges a trained model whose
// forecasts miss by a different amount on alternate held-out records and reads
// the verdict as the scorecard does. Each model's summary counts the held-out
// records as its cases, and its noise is the standard error of the mean of its
// records' scores: nothing for the base model, which misses every record
// alike, and the spread of the alternating misses for the trained one. One
// record, or none, has no such error.
func TestHeldoutVerdictSummaryCountsItsRecords(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	forecast := newHeldoutFixture(t, trainingprogram.MetricForecastMAE)
	commit := func(key string, stage func(string) (artifact.Batch, error), declared ...artifact.ID) {
		t.Helper()
		batch, err := stage(key)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range declared {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id})
		}
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	commit("test-split", func(key string) (artifact.Batch, error) { return forecast.split.PublicationBatch(key, nil) }, forecast.objective.Dataset)
	commit("test-objective", func(key string) (artifact.Batch, error) {
		content, err := forecast.objective.Content()
		return artifact.Batch{Key: key, Contents: []artifact.Content{content}}, err
	})
	session := heldoutTraining(t, store, forecast.objective.Dataset, forecast.view.TrainingMembership)
	base := forecast.score(t, 2, "falling")
	// The trained model misses by a quarter on one record and by three
	// quarters on the next.
	var observations []HeldoutObservation
	for index, target := range forecast.cases {
		miss := 0.25 + 0.5*float64(index%2)
		values := make([]float64, len(target.Values))
		for at, value := range target.Values {
			values[at] = value + miss
		}
		observations = append(observations, HeldoutObservation{Record: target.Record, Values: values, Label: "falling"})
	}
	trained, err := forecast.plan.Score(observations)
	if err != nil {
		t.Fatal(err)
	}
	commit("test-view", forecast.view.Batch, forecast.objective.Processors...)
	commit("test-plan", forecast.plan.Batch)
	commit("test-base", base.Batch)
	commit("test-trained", trained.Batch)
	verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, trained, session)
	if err != nil {
		t.Fatal(err)
	}
	commit("test-verdict", verdict.Batch)

	records := trained.RecordScores()
	if len(records) != len(forecast.cases) || len(records) < 2 {
		t.Fatalf("the trained report keeps %d record scores for %d held-out cases", len(records), len(forecast.cases))
	}
	var mean, squares float64
	for _, record := range records {
		mean += record.Value / float64(len(records))
	}
	for _, record := range records {
		squares += (record.Value - mean) * (record.Value - mean)
	}
	want := math.Sqrt(squares / float64(len(records)-1) / float64(len(records)))
	visited := 0
	err = VisitHeldoutVerdicts(ctx, store, func(_ artifact.ID, before, after EvalSummary) error {
		visited++
		if before.Cases != uint64(len(records)) || after.Cases != before.Cases {
			t.Errorf("the verdict's summaries count %d and %d cases, want its %d held-out records", before.Cases, after.Cases, len(records))
		}
		if before.Noise != 0 || want == 0 || math.Abs(after.Noise-want) > want*1e-12 {
			t.Errorf("the summaries' noise reads %g for the base and %g for the trained model, want none and %g", before.Noise, after.Noise, want)
		}
		return nil
	}, nil)
	if err != nil || visited != 1 {
		t.Fatalf("visited %d verdicts: %v", visited, err)
	}
	if lone := meanError(records[:1]); lone != 0 || meanError(nil) != 0 {
		t.Fatalf("one record reads a standard error of %g", lone)
	}
}
