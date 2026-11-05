package trainingworkflow

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/hostoptimizer"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
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
	store, err := overgodb.Create(filepath.Join(root, "store"))
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
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
	counting.Load = 3 * time.Millisecond
	counting.Train = func(ctx context.Context, recipe artifact.ID, shown []HeldoutExample) (trainingprogram.Checkpoint, error) {
		shownCount = len(shown)
		return run.Train(ctx, recipe, shown)
	}
	result, err := RunHeldout(ctx, store, counting)
	testcheck.NoError(t, err)
	if shownCount != len(training.Records) || !result.Passed {
		t.Fatalf("training saw %d of %d training records, verdict passed %t", shownCount, len(training.Records), result.Passed)
	}
	// The session records the load its caller made for it.
	session, err := runrecord.RequireServingObservation(ctx, store, result.Observation)
	if want := []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: uint64(counting.Load)}}; err != nil || !slices.Equal(session.Phases, want) {
		t.Fatalf("session phases = (%+v, %v), want the run's load %+v", session.Phases, err, want)
	}
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{result.Verdict})
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(parameters)
	testcheck.NoError(t, err)
	weights := make([]float32, parameters)
	muon, err := hostoptimizer.New(weights, slices.Repeat([]float32{1}, parameters), plan, config)
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	if _, err := RecordCheckpoint(t.Context(), store, directory, checkpoint); err != nil {
		t.Fatal(err)
	}
	return checkpoint
}

// TestAnObjectiveRunChargesExposureBeforeItReads: a trained candidate's read of the held-out records is on record before the
// first record is handed over, each candidate is charged once, and a second candidate on the same set reads as the second.
func TestAnObjectiveRunChargesExposureBeforeItReads(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Create(filepath.Join(root, "store"))
	testcheck.NoError(t, err)
	defer store.Close()
	derived := derivedIdentity(t, "overgo/heldout-charge-test/")
	source := derived(artifact.KindDataset, "dataset")
	training, heldout := commitSeriesSplit(t, ctx, store, source, 12)
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "forecast-charge", Kind: trainingprogram.ObjectiveForecast,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityTimeSeries},
			Outputs: []recipecontract.Modality{recipecontract.ModalityTimeSeries},
		},
		Dataset: source, Split: training.ID, Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss: derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricForecastMAE, Evidence: []artifact.ID{derived(artifact.KindEvidence, "note")},
		Authority: trainingprogram.ObjectiveDeclared,
	})
	testcheck.NoError(t, err)
	const alias = "objective.registered.forecast-charge"
	modelPath, modelID := registerObjective(t, ctx, store, root, objective, alias)
	examples := map[string]HeldoutExample{}
	for _, record := range slices.Concat(training.Records, heldout.Records) {
		examples[record.ID] = HeldoutExample{Input: []float64{5, 5, 5}, Target: []float64{5, 5}}
	}
	level, trained := 0.0, false
	candidate := func() trainingprogram.Checkpoint { return recordedCheckpoint(t, store, modelID, source, training.ID) }
	var chargedAtRead []uint64
	run := HeldoutRun{
		Alias: alias, ModelPath: modelPath, Host: true,
		Data: ObjectiveData{Examples: examples, Training: derived(artifact.KindProfile, "training stream"), Heldout: derived(artifact.KindProfile, "held-out stream")},
		Train: func(context.Context, artifact.ID, []HeldoutExample) (trainingprogram.Checkpoint, error) {
			level, trained = 5, true
			return candidate(), nil
		},
		Predict: func(HeldoutExample) (HeldoutPrediction, error) {
			if trained {
				reads, err := evaluation.ChargeHeldoutRead(ctx, store, objective, heldout.ID, derived(artifact.KindCheckpoint, "another candidate"))
				testcheck.NoError(t, err)
				chargedAtRead = append(chargedAtRead, reads)
				trained = false
			}
			return HeldoutPrediction{Values: []float64{level, level}}, nil
		},
	}
	result, err := RunHeldout(ctx, store, run)
	testcheck.NoError(t, err)
	// The stand-in read charged its own candidate after the run's: two, so the run's came first.
	if result.Reads != 1 || !slices.Equal(chargedAtRead, []uint64{2}) {
		t.Fatalf("the run read as %d, and a candidate charged during its read as %v", result.Reads, chargedAtRead)
	}
	if reads, err := evaluation.ChargeHeldoutRead(ctx, store, objective, heldout.ID, result.Trained); err != nil || reads != 2 {
		t.Fatalf("the same candidate charged again = %d, %v; want no new charge", reads, err)
	}
	if reads, err := evaluation.ChargeHeldoutRead(ctx, store, objective, heldout.ID, derived(artifact.KindCheckpoint, "a third candidate")); err != nil || reads != 3 {
		t.Fatalf("a third candidate = %d, %v; want the third read of the set", reads, err)
	}
}
