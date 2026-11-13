//go:build modeltest

package main

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modeltest"
	"overgo/internal/overgodbtest"
	"overgo/internal/runrecord"
	"overgo/internal/tabularicl"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

const (
	// tabfmRows is the directory, beneath the datasets root, of the real table
	// the TabFM proof trains on: credit-g's thousand customers, each a good or
	// a bad credit risk, in the requests whose rows are encoded once for the
	// whole table (their 520 context rows are the same values in all 16). The
	// BeyondArena task metadata the TabFM release carries had held 142
	// datasets, whose held-out half made 26 requests, too few to tell the
	// head's gain of 0.0769 from the 0.0991 they could (tabfm-cases/do).
	tabfmRows = "credit_g_requests_context"
	// tabfmPools is how many groups the customers fall into by their row: a
	// split by group keeps every held-out request's rows off the training ones.
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
	table := filepath.Join(testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t).Datasets, tabfmRows)
	if _, err := os.Stat(table); err != nil {
		testskip.NotApplicable(t, "the datasets root here holds no credit-g requests")
	}
	tables, groups := tabfmRequests(t, table)
	ctx := t.Context()
	root := t.TempDir()
	store := overgodbtest.Create(t, filepath.Join(root, "store"))
	registerTableSplit(t, ctx, store, root, tables, groups, trainingprogram.MetricTableAccuracy)
	result := testcheck.Must(trainTableObjective(ctx, store, tableObjectiveAlias, model, filepath.Join(root, "trained"), tabfmSteps, artifact.ID{}, false))(t)
	// The session records how long the head, the requests and the trainer took to load.
	session, err := runrecord.RequireServingObservation(ctx, store, result.Observation)
	if err != nil || !slices.ContainsFunc(session.Phases, func(phase runrecord.PhaseMetric) bool {
		return phase.Phase == runrecord.PhaseLoad && phase.DurationNS != 0
	}) {
		t.Fatalf("the training session records phases %+v (%v), want a load phase", session.Phases, err)
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
	t.Logf("TabFM label accuracy on the held-out requests' query rows (%d requests split in all): %.4f -> %.4f, a gain of %.4f over %d cases against the %.4f they can tell from noise (verdict %s)",
		len(tables), verdict.BaselineValue, verdict.CandidateValue, result.Gain.Gain, result.Gain.Cases, result.Gain.Detectable, result.Verdict)
	if !result.Passed {
		t.Fatalf("the trained TabFM head's held-out gain was not approved: gain %.4f over %d cases against the %.4f these cases can tell from noise (supported %t)",
			result.Gain.Gain, result.Gain.Cases, result.Gain.Detectable, result.Gain.Supported)
	}
	if _, err := trainingworkflow.PromoteObjectiveAdaptive(ctx, store, tableObjectiveAlias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := trainingworkflow.PromoteObjectiveApproved(ctx, store, tableObjectiveAlias, []artifact.ID{result.Verdict})
	if err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}
}

// tabfmRequests reads each of the table's rows once from the request files in
// directory and turns the rows of each pool into classification requests:
// every rotation of the pool's rows by the query rows is one request, with
// the table's columns, categorical columns and class ids. A row's pool is its
// values' hash; groups names each request's pool.
func tabfmRequests(t *testing.T, directory string) ([]tabularicl.Request, []string) {
	t.Helper()
	type row struct {
		features []float32
		label    float32
	}
	pools := make([][]row, tabfmPools)
	seen := map[string]bool{}
	var table tabularicl.Request
	for _, file := range testcheck.Must(filepath.Glob(filepath.Join(directory, "*.json")))(t) {
		var read tabularicl.Request
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(file))(t), &read))
		if table.Cols == 0 {
			table = read
		}
		if read.Cols != table.Cols || !slices.Equal(read.CatCols, table.CatCols) {
			t.Fatalf("%s holds %d columns, categorical %v, where the table's first request held %d, %v", file, read.Cols, read.CatCols, table.Cols, table.CatCols)
		}
		for index := range read.Rows {
			features := read.X[index*read.Cols : (index+1)*read.Cols]
			key := fmt.Sprint(features)
			if seen[key] {
				continue
			}
			seen[key] = true
			hash := fnv.New32a()
			hash.Write([]byte(key))
			pool := hash.Sum32() % tabfmPools
			pools[pool] = append(pools[pool], row{features: features, label: read.Y[index]})
		}
	}
	rowsPerRequest := tabfmContextRows + tabfmQueryRows
	var tables []tabularicl.Request
	var groups []string
	for pool, members := range pools {
		for start := 0; len(members) >= rowsPerRequest && start < len(members); start += tabfmQueryRows {
			request := tabularicl.Request{
				Task: tabularicl.TaskClassification, Rows: rowsPerRequest, Cols: table.Cols, TrainRows: tabfmContextRows, CatCols: table.CatCols,
			}
			for offset := range rowsPerRequest {
				member := members[(start+offset)%len(members)]
				request.X = append(request.X, member.features...)
				request.Y = append(request.Y, member.label)
			}
			tables, groups = append(tables, request), append(groups, fmt.Sprintf("pool-%d", pool))
		}
	}
	return tables, groups
}
