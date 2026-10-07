package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/tabularicl"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// tableObjectiveAlias names the table-prediction objective each test
// registers in its own store.
const tableObjectiveAlias = "objective.registered.tables"

// tableSignature is the table pair a table-prediction objective trains.
var tableSignature = recipecontract.ModalitySignature{
	Inputs:  []recipecontract.Modality{recipecontract.ModalityTable},
	Outputs: []recipecontract.Modality{recipecontract.ModalityTable},
}

// TestTabularObjectiveApprovesOnLabelAccuracyGain walks a registered
// dataset of TabFM requests through a table-prediction objective: its
// examples come through the training data path keyed by record, with each
// held-out request's query-row classes as the target, and a head that
// learns the classes climbs the objective from declared to approved on its
// held-out label accuracy.
func TestTabularObjectiveApprovesOnLabelAccuracyGain(t *testing.T) {
	t.Parallel()
	var tables []tabularicl.Request
	for table := range 12 {
		tables = append(tables, tabularicl.Request{
			Task: tabularicl.TaskClassification, Rows: 4, Cols: 1, TrainRows: 2,
			X: []float32{float32(table), 1, 2, 3}, Y: []float32{0, 1, float32(table % 2), 1},
		})
	}
	// A head that has not learned answers each query row with the other
	// class, matching none; the trained head matches all.
	approveTableObjective(t, trainingprogram.MetricTableAccuracy, tables, func(class float32) float32 { return 1 - class }, 0, 1)
}

// TestTrainTabularRegressionObjective walks a registered dataset of TabFM
// regression requests through a table-prediction objective judged by the
// query rows' mean squared error: the objective's metric names the regression
// head, each held-out request's query-row values are the target, and a head
// that closes the error climbs the objective from declared to approved on
// its held-out verdict.
func TestTrainTabularRegressionObjective(t *testing.T) {
	t.Parallel()
	var tables []tabularicl.Request
	for table := range 12 {
		tables = append(tables, tabularicl.Request{
			Task: tabularicl.TaskRegression, Rows: 4, Cols: 1, TrainRows: 2,
			X: []float32{float32(table), 1, 2, 3}, Y: []float32{0.5, 1.5, float32(table) / 4, 2},
		})
	}
	// A head that has not learned answers each query row two units high, a
	// squared error of 4 where the absolute error would be 2; the trained
	// head answers exactly.
	approveTableObjective(t, trainingprogram.MetricTableMSE, tables, func(value float32) float32 { return value + 2 }, 4, 0)
}

// approveTableObjective registers the tables as a dataset, splits it, and
// registers a table-prediction objective judged by metric over its training
// membership; it reads the objective's requests for the head the metric
// names, trains through a held-out run that publishes a real checkpoint, and
// holds the verdict to measuring metric as base and trained, and the
// objective to climbing from declared to approved on it. Before training the
// head answers each query row's target through untrained, after it exactly.
func approveTableObjective(t *testing.T, metric trainingprogram.EvaluationMetric, tables []tabularicl.Request, untrained func(float32) float32, base, trained float64) {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	objective, modelPath := registerTableSplit(t, ctx, store, root, tables, nil, metric)

	task := tableObjectiveTasks[metric]
	if task != tables[0].Task {
		t.Fatalf("metric %q trains the %q head, want %q", metric, task, tables[0].Task)
	}
	requests, data, err := objectiveRequests(ctx, store, objective, task)
	if err != nil || !data.Training.Valid() || !data.Heldout.Valid() {
		t.Fatalf("objective requests = (streams %s and %s, %v), want both memberships' streams", data.Training, data.Heldout, err)
	}
	if len(requests) != len(tables) || len(data.Examples) != len(tables) {
		t.Fatalf("read %d requests and %d examples, want all %d tables", len(requests), len(data.Examples), len(tables))
	}
	learned := false
	result, err := trainingworkflow.RunHeldout(ctx, store, trainingworkflow.HeldoutRun{
		Alias: tableObjectiveAlias, ModelPath: modelPath, Data: data, Host: true,
		Train: func(ctx context.Context, recipeID artifact.ID, _ []trainingworkflow.HeldoutExample) (trainingprogram.Checkpoint, error) {
			learned = true
			return publishUnderObjective(t, ctx, store, recipeID, trainingprogram.ObjectiveTablePrediction, data.Training)
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			request := requests[example.Record]
			values := make([]float64, 0, request.Rows-request.TrainRows)
			for _, target := range request.Y[request.TrainRows:] {
				if !learned {
					target = untrained(target)
				}
				values = append(values, float64(target))
			}
			return trainingworkflow.HeldoutPrediction{Values: values}, nil
		},
	})
	if err != nil || !result.Passed {
		t.Fatalf("held-out run = (passed=%t, %v), want a passed verdict", result.Passed, err)
	}
	var verdict struct {
		Metric         trainingprogram.EvaluationMetric `json:"metric"`
		BaselineValue  float64                          `json:"baseline_value"`
		CandidateValue float64                          `json:"candidate_value"`
	}
	_, content, found, err := store.OpenContent(ctx, result.Verdict)
	if err != nil || !found {
		t.Fatalf("verdict %s content = (found %t, %v)", result.Verdict, found, err)
	}
	document, err := io.ReadAll(content)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(document, &verdict); err != nil || verdict.Metric != metric || verdict.BaselineValue != base || verdict.CandidateValue != trained {
		t.Fatalf("verdict = %+v (%v), want %s from %g to %g", verdict, err, metric, base, trained)
	}
	if _, err := trainingworkflow.PromoteObjectiveAdaptive(ctx, store, tableObjectiveAlias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := trainingworkflow.PromoteObjectiveApproved(ctx, store, tableObjectiveAlias, []artifact.ID{result.Verdict})
	if err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}
}

// registerTableSplit registers the tables as a dataset, one request file per
// record, splits it in halves by group, groups[i] being table i's group or,
// for nil groups, each record its own, and registers a table-prediction
// objective judged by metric over its training membership.
func registerTableSplit(t *testing.T, ctx context.Context, store *overgodb.Store, root string, tables []tabularicl.Request, groups []string, metric trainingprogram.EvaluationMetric) (trainingprogram.ObjectiveDocument, string) {
	t.Helper()
	corpus := t.TempDir()
	for table, request := range tables {
		data, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		// Names that sort in table order keep record i table i.
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("table-%04d.json", table)), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	source, training := registerSplit(t, ctx, store, "tables", corpus, len(tables), groups)
	return registerTableObjective(t, ctx, store, root, source, training, metric)
}

// registerSplit registers the count files of corpus as a dataset, one record
// a file in name order, and splits it in halves by group, groups[i] being
// record i's group or, for nil groups, each record its own. It returns the
// dataset and its training membership.
func registerSplit(t *testing.T, ctx context.Context, store *overgodb.Store, name, corpus string, count int, groups []string) (source, training artifact.ID) {
	t.Helper()
	registered, err := dataset.RegisterDirectoryDataset(ctx, store, name, corpus)
	if err != nil {
		t.Fatal(err)
	}
	var records []dataset.Record
	for index := range uint64(count) {
		id := dataset.RecordID(registered.Dataset, dataset.InventoryAssetName, index)
		group := id
		if groups != nil {
			group = groups[index]
		}
		records = append(records, dataset.Record{ID: id, Group: group})
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
	for _, membership := range split.Memberships {
		if membership.Partition == "train" {
			training = membership.ID
		}
	}
	return registered.Dataset, training
}

// registerTableObjective registers a table-prediction objective judged by
// metric over the dataset's training membership, and writes a head file to
// record.
func registerTableObjective(t *testing.T, ctx context.Context, store *overgodb.Store, root string, source, training artifact.ID, metric trainingprogram.EvaluationMetric) (trainingprogram.ObjectiveDocument, string) {
	t.Helper()
	return registerObjective(t, ctx, store, root, tableObjectiveAlias, trainingprogram.ObjectiveSpec{
		Name: "tables", Kind: trainingprogram.ObjectiveTablePrediction, Signature: tableSignature,
		Dataset: source, Split: training, Metric: metric,
	})
}

// registerObjective registers, under alias, the declared objective spec
// names over its dataset's training membership, with stand-in descriptors
// for the contract and bootstrap identities it names, and writes a weights
// file to record.
func registerObjective(t *testing.T, ctx context.Context, store *overgodb.Store, root, alias string, spec trainingprogram.ObjectiveSpec) (trainingprogram.ObjectiveDocument, string) {
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
	spec.Processors = []artifact.ID{stand(artifact.KindProfile, "overgo/objective-test/processor")}
	spec.Loss, spec.Evaluation = stand(artifact.KindProfile, "overgo/objective-test/loss"), stand(artifact.KindProfile, "overgo/objective-test/evaluation")
	spec.Evidence, spec.Authority = []artifact.ID{stand(artifact.KindEvidence, "overgo/objective-test/note")}, trainingprogram.ObjectiveDeclared
	objective, err := trainingprogram.NewObjective(spec)
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
		Aliases: []artifact.AliasBinding{{Name: alias, Target: objective.ID}},
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
