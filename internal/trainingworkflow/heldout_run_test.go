package trainingworkflow

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
)

// TestHeldoutRunTrainsOnlyOnTrainingRecords walks the held-out chain with a
// one-parameter forecaster: training sees exactly the objective's training
// membership, the base and trained models are judged on the held-out
// membership, and the published verdict climbs the objective to approved.
// A held-out record with no example is refused rather than skipped.
func TestHeldoutRunTrainsOnlyOnTrainingRecords(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	derived := derivedIdentity(t, "overgo/heldout-run-test/")
	source := derived(artifact.KindDataset, "dataset")
	training, heldout := commitSeriesSplit(t, ctx, store, source, 12)
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "forecast-run", Kind: trainingprogram.ObjectiveForecast,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityTimeSeries},
			Outputs: []recipecontract.Modality{recipecontract.ModalityTimeSeries},
		},
		Dataset: source, Split: training.ID, Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss: derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricForecastMAE, Evidence: []artifact.ID{derived(artifact.KindEvidence, "note")},
		Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	const alias = "objective.registered.forecast-run"
	modelPath, _ := registerObjective(t, ctx, store, root, objective, alias)

	// Every series continues at level 5; the forecaster starts at 0 and
	// training moves it to the mean target of what it is shown.
	examples := map[string]HeldoutExample{}
	for _, record := range slices.Concat(training.Records, heldout.Records) {
		examples[record.ID] = HeldoutExample{Input: []float64{5, 5, 5}, Target: []float64{5, 5}}
	}
	level := 0.0
	run := HeldoutRun{
		Alias: alias, ModelPath: modelPath, Examples: examples, Host: true,
		Train: func(_ context.Context, shown []HeldoutExample) error {
			var sum float64
			for _, example := range shown {
				sum += example.Target[0]
			}
			level = sum / float64(len(shown))
			return nil
		},
		Predict: func(example HeldoutExample) (HeldoutPrediction, error) {
			return HeldoutPrediction{Values: []float64{level, level}}, nil
		},
	}
	missing := run
	missing.Examples = map[string]HeldoutExample{}
	for id, example := range examples {
		if id != heldout.Records[0].ID {
			missing.Examples[id] = example
		}
	}
	if _, err := RunHeldout(ctx, store, missing); err == nil || !strings.Contains(err.Error(), "has no example") {
		t.Fatalf("a held-out record with no example = %v, want it refused", err)
	}

	// Train must see exactly the training membership's examples.
	shownCount := 0
	counting := run
	counting.Train = func(ctx context.Context, shown []HeldoutExample) error {
		shownCount = len(shown)
		return run.Train(ctx, shown)
	}
	result, err := RunHeldout(ctx, store, counting)
	if err != nil {
		t.Fatal(err)
	}
	if shownCount != len(training.Records) || !result.Passed {
		t.Fatalf("training saw %d of %d training records, verdict passed %t", shownCount, len(training.Records), result.Passed)
	}
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{result.Verdict})
	if err != nil {
		t.Fatal(err)
	}
	if approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("authority after a passed held-out run = %q", approved.Authority)
	}
}
