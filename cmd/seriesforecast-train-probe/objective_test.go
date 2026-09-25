package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestForecastObjectiveProbeScoresOnlyHeldoutRecords walks a registered
// light-curve dataset through a forecast objective's split: the probe's
// examples come through the training data path keyed by record, training is
// shown only the training membership's curves, and prediction answers only
// the held-out membership's curves, before and after training.
func TestForecastObjectiveProbeScoresOnlyHeldoutRecords(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	const curves, patchLen, horizon = 12, 4, 2
	corpus := t.TempDir()
	for curve := range curves {
		var values []float64
		for sample := range 12 {
			values = append(values, float64(sample), 500, float64(100*curve+sample), 0.1)
		}
		data := fmt.Sprintf(`{"x":%s}`, jsonFloats(values))
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("curve-%02d.json", curve)), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	registered, err := dataset.RegisterDirectoryDataset(ctx, store, "curves", corpus)
	if err != nil {
		t.Fatal(err)
	}
	var records []dataset.Record
	for index := range uint64(curves) {
		id := dataset.RecordID(registered.Dataset, dataset.InventoryAssetName, index)
		records = append(records, dataset.Record{ID: id, Group: id})
	}
	split, err := dataset.BuildGroupSplit(registered.Dataset, records, 17, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	memberships := map[string]dataset.Membership{}
	for _, membership := range split.Memberships {
		memberships[membership.Partition] = membership
	}
	training, heldout := memberships["train"], memberships["heldout"]
	derive := func(kind artifact.Kind, role string) artifact.ID {
		t.Helper()
		id, err := artifact.IdentifyBytes(kind, []byte("overgo/forecast-probe-test/"+role))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "forecast-probe", Kind: trainingprogram.ObjectiveForecast, Signature: forecastSignature,
		Dataset: registered.Dataset, Split: training.ID, Processors: []artifact.ID{derive(artifact.KindProfile, "processor")},
		Loss: derive(artifact.KindProfile, "loss"), Evaluation: derive(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricForecastMAE, Evidence: []artifact.ID{derive(artifact.KindEvidence, "note")},
		Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	const alias = "objective.registered.forecast-probe"
	modelPath := commitForecastObjective(t, ctx, store, root, split, objective, alias)

	examples, err := objectiveExamples(ctx, store, objective, patchLen, horizon)
	if err != nil {
		t.Fatal(err)
	}
	if len(examples) != curves {
		t.Fatalf("materialized %d examples, want all %d curves", len(examples), curves)
	}
	inputs := map[string]string{}
	for id, example := range examples {
		inputs[fmt.Sprint(example.Input)] = id
	}
	partitionOf := func(membership dataset.Membership) map[string]bool {
		ids := map[string]bool{}
		for _, record := range membership.Records {
			ids[record.ID] = true
		}
		return ids
	}
	trainingIDs, heldoutIDs := partitionOf(training), partitionOf(heldout)
	level := 0.0
	var trained, predicted []string
	result, err := trainingworkflow.RunHeldout(ctx, store, trainingworkflow.HeldoutRun{
		Alias: alias, ModelPath: modelPath, Examples: examples, Host: true,
		Train: func(_ context.Context, shown []trainingworkflow.HeldoutExample) error {
			var sum float64
			for _, example := range shown {
				trained = append(trained, inputs[fmt.Sprint(example.Input)])
				sum += example.Target[0]
			}
			level = sum / float64(len(shown))
			return nil
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			predicted = append(predicted, inputs[fmt.Sprint(example.Input)])
			return trainingworkflow.HeldoutPrediction{Values: slices.Repeat([]float64{level}, len(example.Target))}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range trained {
		if !trainingIDs[id] {
			t.Fatalf("training was shown %s, which is not a training record", id)
		}
	}
	for _, id := range predicted {
		if !heldoutIDs[id] {
			t.Fatalf("prediction answered %s, which is not a held-out record", id)
		}
	}
	if len(trained) != len(training.Records) || len(predicted) != 2*len(heldout.Records) || result.Verdict.Kind() != artifact.KindEvaluation {
		t.Fatalf("trained %d of %d, predicted %d of 2x%d, verdict %s", len(trained), len(training.Records), len(predicted), len(heldout.Records), result.Verdict)
	}
}

// commitForecastObjective publishes the split and registers the objective
// under alias with stand-in descriptors for the contract identities it names
// and the bootstrap profiles, writing a model file to record; it returns the
// model file's path.
func commitForecastObjective(t *testing.T, ctx context.Context, store *overgodb.Store, root string, split dataset.SplitPlan, objective trainingprogram.ObjectiveDocument, alias string) string {
	t.Helper()
	splitBatch, err := split.PublicationBatch("test-split", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, splitBatch); err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(root, "model.safetensors")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	referenced := []artifact.ID{objective.Loss, objective.Evaluation}
	referenced = append(referenced, objective.Processors...)
	referenced = append(referenced, objective.Evidence...)
	for _, name := range []string{"precision", "placement", "memory", "checkpoint", "promotion"} {
		id, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training-bootstrap/"+name))
		if err != nil {
			t.Fatal(err)
		}
		referenced = append(referenced, id)
	}
	descriptors := make([]artifact.Descriptor, 0, len(referenced))
	for _, id := range referenced {
		descriptors = append(descriptors, artifact.Descriptor{ID: id})
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "test-objective", Contents: []artifact.Content{content}, Artifacts: descriptors,
		Aliases: []artifact.AliasBinding{{Name: alias, Target: objective.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	return modelPath
}

func jsonFloats(values []float64) string {
	text := "["
	for index, value := range values {
		if index > 0 {
			text += ","
		}
		text += fmt.Sprint(value)
	}
	return text + "]"
}
