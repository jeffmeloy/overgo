package trainingworkflow

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
	"overgo/internal/optimizer"
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
	modelPath, modelID := registerObjective(t, ctx, store, root, objective, alias)

	// Every series continues at level 5; the forecaster starts at 0 and
	// training moves it to the mean target of what it is shown.
	examples := map[string]HeldoutExample{}
	for _, record := range slices.Concat(training.Records, heldout.Records) {
		examples[record.ID] = HeldoutExample{Input: []float64{5, 5, 5}, Target: []float64{5, 5}}
	}
	level := 0.0
	data := ObjectiveData{Examples: examples, Training: derived(artifact.KindProfile, "training stream"), Heldout: derived(artifact.KindProfile, "held-out stream")}
	run := HeldoutRun{
		Alias: alias, ModelPath: modelPath, Data: data, Host: true,
		Train: func(_ context.Context, _ artifact.ID, shown []HeldoutExample) (trainingprogram.Checkpoint, error) {
			var sum float64
			for _, example := range shown {
				sum += example.Target[0]
			}
			level = sum / float64(len(shown))
			return recordedCheckpoint(t, store, modelID, source, training.ID), nil
		},
		Predict: func(example HeldoutExample) (HeldoutPrediction, error) {
			return HeldoutPrediction{Values: []float64{level, level}}, nil
		},
	}
	missing := run
	missing.Data.Examples = map[string]HeldoutExample{}
	for id, example := range examples {
		if id != heldout.Records[0].ID {
			missing.Data.Examples[id] = example
		}
	}
	if _, err := RunHeldout(ctx, store, missing); err == nil || !strings.Contains(err.Error(), "has no example") {
		t.Fatalf("a held-out record with no example = %v, want it refused", err)
	}

	// Train must see exactly the training membership's examples.
	shownCount := 0
	counting := run
	counting.Train = func(ctx context.Context, recipe artifact.ID, shown []HeldoutExample) (trainingprogram.Checkpoint, error) {
		shownCount = len(shown)
		return run.Train(ctx, recipe, shown)
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

// recordedCheckpoint publishes one host Muon step of a stand-in matrix as
// the checkpoint of model trained on split of source, under stand-in run
// authority, and records it in store.
func recordedCheckpoint(t *testing.T, store *overgodb.Store, model, source, split artifact.ID) trainingprogram.Checkpoint {
	t.Helper()
	derived := derivedIdentity(t, "overgo/recorded-checkpoint-test/")
	const parameters = 4
	plan, err := hostoptimizer.CompilePlan(parameters, []hostoptimizer.GroupSpec{{Name: "weight", End: parameters, Rows: 2, Cols: 2}})
	if err != nil {
		t.Fatal(err)
	}
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(parameters)
	if err != nil {
		t.Fatal(err)
	}
	weights := make([]float32, parameters)
	muon, err := hostoptimizer.New(weights, slices.Repeat([]float32{1}, parameters), plan, config)
	if err != nil {
		t.Fatal(err)
	}
	muon.Step()
	state := muon.Snapshot()
	rng := derived(artifact.KindProfile, "rng")
	directory := filepath.Join(t.TempDir(), "checkpoint")
	checkpoint, err := trainingprogram.PublishCheckpoint(directory, trainingprogram.CheckpointSpec{
		RunPlan: derived(artifact.KindRecipe, "run plan"), Program: derived(artifact.KindRecipe, "program"), Model: model,
		Dataset: source, Split: split, Stream: trainingprogram.DatasetState{Identity: derived(artifact.KindProfile, "stream"), Position: 1},
		Optimizer: state, ParameterCount: state.Parameters(),
		RNG:        []trainingprogram.RNGState{{Name: "augmentation", Algorithm: rng}, {Name: "data", Algorithm: rng}},
		Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Lineage:    []trainingprogram.LineageParent{{Artifact: model, Relation: artifact.RelationTrainedFrom}},
	}, func(stage string) error {
		return optimizer.SavePlanWeights(filepath.Join(stage, trainingprogram.CheckpointWeights), plan, weights)
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RecordCheckpoint(t.Context(), store, directory, checkpoint); err != nil {
		t.Fatal(err)
	}
	return checkpoint
}
