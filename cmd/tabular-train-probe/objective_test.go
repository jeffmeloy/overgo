package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/tabularicl"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestTabularObjectiveProbeApprovesOnLabelAccuracyGain walks a registered
// dataset of TabFM requests through a table-prediction objective: its
// examples come through the training data path keyed by record, with each
// held-out request's query-row classes as the target, and a head that
// learns the classes climbs the objective from declared to approved on its
// held-out label accuracy.
func TestTabularObjectiveProbeApprovesOnLabelAccuracyGain(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	const tables = 12
	corpus := t.TempDir()
	for table := range tables {
		request := tabularicl.Request{
			Task: "classification", Rows: 4, Cols: 1, TrainRows: 2,
			X: []float32{float32(table), 1, 2, 3}, Y: []float32{0, 1, float32(table % 2), 1},
		}
		data, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("table-%02d.json", table)), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	registered, err := dataset.RegisterDirectoryDataset(ctx, store, "tables", corpus)
	if err != nil {
		t.Fatal(err)
	}
	var records []dataset.Record
	for index := range uint64(tables) {
		id := dataset.RecordID(registered.Dataset, dataset.InventoryAssetName, index)
		records = append(records, dataset.Record{ID: id, Group: id})
	}
	split, err := dataset.BuildGroupSplit(registered.Dataset, records, 17, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	splitBatch, err := split.PublicationBatch("test-split", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, splitBatch); err != nil {
		t.Fatal(err)
	}
	var training artifact.ID
	for _, membership := range split.Memberships {
		if membership.Partition == "train" {
			training = membership.ID
		}
	}
	objective, modelPath := registerTableObjective(t, ctx, store, root, registered.Dataset, training)
	const alias = "objective.registered.tables"

	requests, examples, err := objectiveRequests(ctx, store, objective, "classification")
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != tables || len(examples) != tables {
		t.Fatalf("read %d requests and %d examples, want all %d tables", len(requests), len(examples), tables)
	}
	learned := false
	result, err := trainingworkflow.RunHeldout(ctx, store, trainingworkflow.HeldoutRun{
		Alias: alias, ModelPath: modelPath, Examples: examples, Host: true,
		Train: func(context.Context, []trainingworkflow.HeldoutExample) error { learned = true; return nil },
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			request := requests[example.Record]
			classes := make([]float64, 0, request.Rows-request.TrainRows)
			for _, class := range request.Y[request.TrainRows:] {
				if !learned {
					class = 1 - class
				}
				classes = append(classes, float64(class))
			}
			return trainingworkflow.HeldoutPrediction{Values: classes}, nil
		},
	})
	if err != nil || !result.Passed {
		t.Fatalf("held-out run = (passed=%t, %v), want a passed verdict", result.Passed, err)
	}
	if _, err := trainingworkflow.PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := trainingworkflow.PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{result.Verdict})
	if err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}
}

// registerTableObjective registers a table-prediction objective over the
// dataset's training membership, with stand-in descriptors for the contract
// and bootstrap identities it names, and writes a head file to record.
func registerTableObjective(t *testing.T, ctx context.Context, store *overgodb.Store, root string, source, training artifact.ID) (trainingprogram.ObjectiveDocument, string) {
	t.Helper()
	var referenced []artifact.ID
	stand := func(kind artifact.Kind, name string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(name))
		if err != nil {
			t.Fatal(err)
		}
		referenced = append(referenced, id)
		return id
	}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "tables", Kind: trainingprogram.ObjectiveTablePrediction, Signature: tableSignature,
		Dataset: source, Split: training, Processors: []artifact.ID{stand(artifact.KindProfile, "overgo/tabular-test/processor")},
		Loss: stand(artifact.KindProfile, "overgo/tabular-test/loss"), Evaluation: stand(artifact.KindProfile, "overgo/tabular-test/evaluation"),
		Metric: trainingprogram.MetricTableAccuracy, Evidence: []artifact.ID{stand(artifact.KindEvidence, "overgo/tabular-test/note")},
		Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"precision", "placement", "memory", "checkpoint", "promotion"} {
		stand(artifact.KindProfile, "overgo/training-bootstrap/"+name)
	}
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch := artifact.Batch{
		Key: "test-objective", Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{{Name: "objective.registered.tables", Target: objective.ID}},
	}
	for _, id := range referenced {
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id})
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(root, "head.safetensors")
	if err := os.WriteFile(modelPath, []byte("head"), 0o600); err != nil {
		t.Fatal(err)
	}
	return objective, modelPath
}
