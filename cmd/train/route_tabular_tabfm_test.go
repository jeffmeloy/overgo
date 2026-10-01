//go:build modeltest

package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/tabularicl"
	"overgo/internal/testskip"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// tabfmTable is the real table the TabFM proof trains on: the BeyondArena
// task metadata the TabFM release carries beside its heads, one row per
// task split of a benchmark dataset.
var tabfmTable = filepath.Join("tabarena", "packages", "tabarena", "src", "tabarena", "benchmark", "task", "metadata", "sources", "data", "BeyondArena_tasks_metadata.csv")

// tabfmFeatures are the numeric columns a dataset is described by, and
// tabfmLabel the column its class is: whether its task is binary or
// multiclass classification or regression.
var tabfmFeatures = []string{
	"num_instances", "num_features", "num_classes", "num_instance_groups", "num_text_cols",
	"num_high_cardinality_cats", "num_cols_after_preprocessing", "missing_value_fraction",
}

const (
	tabfmLabel = "problem_type"
	// tabfmPools is how many groups the datasets fall into by name: a split
	// by group keeps every held-out request's rows off the training ones.
	tabfmPools = 8
	// tabfmContextRows and tabfmQueryRows shape each request: the rows whose
	// classes the head reads, then the rows it answers.
	tabfmContextRows = 8
	tabfmQueryRows   = 4
	// tabfmSteps is how many decoder updates the objective trains.
	tabfmSteps = 40
)

// TestTabFMObjectiveApprovesOnHeldOutRows proves held-out approval on the
// real TabFM-1.0 classification head: a table-prediction objective over a
// real table, split by group, trains the head's decoder through the tabular
// route on the training membership's requests, and the verdict on the
// held-out requests' query rows must show a label-accuracy gain that climbs
// the objective to approved. Its one run is the row's acceptance and the
// model's training evidence.
func TestTabFMObjectiveApprovesOnHeldOutRows(t *testing.T) {
	testskip.Short(t, "trains the real TabFM classification head")
	model := modeltest.Directory(t, "tabfm-1.0.0-pytorch")
	table := filepath.Join(model, tabfmTable)
	if _, err := os.Stat(table); err != nil {
		testskip.NotApplicable(t, "the TabFM release here carries no BeyondArena task table")
	}
	tables, groups := tabfmRequests(t, table)
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	registerTableSplit(t, ctx, store, root, tables, groups, trainingprogram.MetricTableAccuracy)
	result, err := trainTableObjective(ctx, store, tableObjectiveAlias, model, filepath.Join(root, "trained"), tabfmSteps)
	if err != nil {
		t.Fatal(err)
	}
	var verdict struct {
		BaselineValue  float64 `json:"baseline_value"`
		CandidateValue float64 `json:"candidate_value"`
	}
	_, content, found, err := store.OpenContent(ctx, result.Verdict)
	if err != nil || !found {
		t.Fatalf("verdict %s content = (found %t, %v)", result.Verdict, found, err)
	}
	data, err := io.ReadAll(content)
	if err != nil || json.Unmarshal(data, &verdict) != nil {
		t.Fatalf("verdict %s = %v", result.Verdict, err)
	}
	t.Logf("TabFM label accuracy on the held-out requests' query rows (%d requests split in all): %.4f -> %.4f (verdict %s)", len(tables), verdict.BaselineValue, verdict.CandidateValue, result.Verdict)
	if !result.Passed {
		t.Fatal("the trained TabFM head did not gain held-out label accuracy")
	}
	if _, err := trainingworkflow.PromoteObjectiveAdaptive(ctx, store, tableObjectiveAlias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := trainingworkflow.PromoteObjectiveApproved(ctx, store, tableObjectiveAlias, []artifact.ID{result.Verdict})
	if err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}
}

// tabfmRequests reads the table's first row for each dataset and turns the
// datasets of each pool into classification requests: every rotation of the
// pool's rows by the query rows is one request, its features signed log
// magnitudes and its classes the label's values in name order. groups names
// each request's pool.
func tabfmRequests(t *testing.T, path string) ([]tabularicl.Request, []string) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	rows, err := csv.NewReader(file).ReadAll()
	if err != nil || len(rows) < 2 {
		t.Fatalf("table %s: %d rows, %v", path, len(rows), err)
	}
	header := rows[0]
	column := func(name string) int {
		index := slices.Index(header, name)
		if index < 0 {
			t.Fatalf("table %s has no %s column", path, name)
		}
		return index
	}
	name, label := column("dataset_name"), column(tabfmLabel)
	type described struct {
		features []float32
		label    string
	}
	pools := make([][]described, tabfmPools)
	seen := map[string]bool{}
	var labels []string
	for _, row := range rows[1:] {
		if seen[row[name]] {
			continue
		}
		seen[row[name]] = true
		var features []float32
		for _, feature := range tabfmFeatures {
			value, err := strconv.ParseFloat(row[column(feature)], 64)
			if err != nil {
				value = 0
			}
			features = append(features, float32(math.Copysign(math.Log1p(math.Abs(value)), value)))
		}
		hash := fnv.New32a()
		hash.Write([]byte(row[name]))
		pool := hash.Sum32() % tabfmPools
		pools[pool] = append(pools[pool], described{features: features, label: row[label]})
		if !slices.Contains(labels, row[label]) {
			labels = append(labels, row[label])
		}
	}
	slices.Sort(labels)
	rowsPerRequest := tabfmContextRows + tabfmQueryRows
	var tables []tabularicl.Request
	var groups []string
	for pool, members := range pools {
		for start := 0; len(members) >= rowsPerRequest && start < len(members); start += tabfmQueryRows {
			request := tabularicl.Request{
				Task: tabularicl.TaskClassification, Rows: rowsPerRequest, Cols: len(tabfmFeatures), TrainRows: tabfmContextRows,
			}
			for offset := range rowsPerRequest {
				member := members[(start+offset)%len(members)]
				request.X = append(request.X, member.features...)
				request.Y = append(request.Y, float32(slices.Index(labels, member.label)))
			}
			tables, groups = append(tables, request), append(groups, fmt.Sprintf("pool-%d", pool))
		}
	}
	return tables, groups
}
