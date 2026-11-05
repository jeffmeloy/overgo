package evaluation

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/hostoptimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// heldoutFixture is a forecast or table objective bound to the training half
// of a real group split, its held-out view, and the held-out plan its metric
// compiles over the held-out records' cases.
type heldoutFixture struct {
	objective trainingprogram.ObjectiveDocument
	view      SFTEvaluationView
	plan      HeldoutPlan
	cases     []HeldoutCase
	split     dataset.SplitPlan
}

// alternatingReading reads each document a difference of 1 and 2 in turn, counting its calls in prepared.
func alternatingReading(prepared *int) func(documents []string) ([]float64, error) {
	return func(documents []string) ([]float64, error) {
		*prepared++
		differences := make([]float64, len(documents))
		for index := range differences {
			differences[index] = 1 + float64(index%2)
		}
		return differences, nil
	}
}

// heldoutCommit publishes, under key, the batch stage gives for it, with the artifacts declared beside it.
type heldoutCommit func(key string, stage func(string) (artifact.Batch, error), declared ...artifact.ID)

// heldoutCommitter commits into store for a test.
func heldoutCommitter(t *testing.T, store *overgodb.Store) heldoutCommit {
	return func(key string, stage func(string) (artifact.Batch, error), declared ...artifact.ID) {
		t.Helper()
		batch, err := stage(key)
		testcheck.NoError(t, err)
		for _, id := range declared {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id})
		}
		if _, err := artifact.Publish(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
	}
}

// publishObjective commits the fixture's split, its dataset declared beside it, and its objective.
func (f heldoutFixture) publishObjective(commit heldoutCommit) {
	commit("test-split", func(key string) (artifact.Batch, error) { return f.split.PublicationBatch(key, nil) }, f.objective.Dataset)
	commit("test-objective", func(key string) (artifact.Batch, error) {
		content, err := f.objective.Content()
		return artifact.Batch{Key: key, Contents: []artifact.Content{content}}, err
	})
}

func newHeldoutFixture(t *testing.T, metric trainingprogram.EvaluationMetric) heldoutFixture {
	t.Helper()
	derived := func(kind artifact.Kind, role string) artifact.ID {
		return testutil.ArtifactID(t, kind, "overgo/heldout-verdict-test/"+role)
	}
	source := derived(artifact.KindDataset, "dataset")
	var records []dataset.Record
	for index := range 12 {
		records = append(records, dataset.Record{ID: fmt.Sprintf("series/%02d", index), Group: fmt.Sprintf("series-%02d", index)})
	}
	split, err := dataset.BuildGroupSplit(source, records, 17, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	testcheck.NoError(t, err)
	var training, heldout dataset.Membership
	for _, membership := range split.Memberships {
		switch membership.Partition {
		case "train":
			training = membership
		case "heldout":
			heldout = membership
		}
	}
	if len(training.Records) == 0 || len(heldout.Records) == 0 {
		t.Fatalf("split left a partition empty: train=%d heldout=%d", len(training.Records), len(heldout.Records))
	}
	kind, modality := trainingprogram.ObjectiveForecast, recipecontract.ModalityTimeSeries
	if metric == trainingprogram.MetricTableAccuracy {
		kind, modality = trainingprogram.ObjectiveTablePrediction, recipecontract.ModalityTable
	}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "heldout-" + string(metric), Kind: kind,
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{modality}, Outputs: []recipecontract.Modality{modality},
		},
		Dataset: source, Split: training.ID, Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss: derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: metric, Evidence: []artifact.ID{derived(artifact.KindEvidence, "observation")},
		Authority: trainingprogram.ObjectiveAdaptive,
	})
	testcheck.NoError(t, err)
	view, err := CompileSFTEvaluationView(objective, training, heldout)
	testcheck.NoError(t, err)
	var cases []HeldoutCase
	for index, record := range view.Records {
		cases = append(cases, HeldoutCase{Record: record.ID, Values: []float64{float64(index), 1}, Label: "rising"})
	}
	plan, err := CompileHeldoutPlan(objective, view, cases)
	testcheck.NoError(t, err)
	return heldoutFixture{objective: objective, view: view, plan: plan, cases: cases, split: split}
}

// heldoutTraining publishes a checkpoint that trained a stand-in model on
// training of source and returns what a verdict binds of it. With a store,
// it commits the checkpoint's record and a succeeded training session
// observing that model; without one, the observation is a stand-in.
func heldoutTraining(t *testing.T, store artifact.Repository, source, training artifact.ID) HeldoutTraining {
	t.Helper()
	ctx := t.Context()
	derived := func(kind artifact.Kind, role string) artifact.ID {
		return testutil.ArtifactID(t, kind, "overgo/heldout-training-test/"+role)
	}
	const parameters = 4
	plan, err := hostoptimizer.CompilePlan(parameters, []hostoptimizer.GroupSpec{{Name: "weight", End: parameters, Rows: 2, Cols: 2}})
	testcheck.NoError(t, err)
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(parameters)
	testcheck.NoError(t, err)
	muon, err := hostoptimizer.New(make([]float32, parameters), slices.Repeat([]float32{1}, parameters), plan, config)
	testcheck.NoError(t, err)
	muon.Step()
	state := muon.Snapshot()
	model, rng := derived(artifact.KindModel, "model"), derived(artifact.KindProfile, "rng")
	directory := filepath.Join(t.TempDir(), "checkpoint")
	checkpoint, err := trainingprogram.PublishCheckpoint(directory, trainingprogram.CheckpointSpec{
		RunPlan: derived(artifact.KindRecipe, "run plan"), Program: derived(artifact.KindRecipe, "program"), Model: model,
		Dataset: source, Split: training, Stream: trainingprogram.DatasetState{Identity: derived(artifact.KindProfile, "stream"), Position: 1},
		Optimizer: state, ParameterCount: state.Parameters(),
		RNG:        []trainingprogram.RNGState{{Name: "augmentation", Algorithm: rng}, {Name: "data", Algorithm: rng}},
		Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Lineage:    []trainingprogram.LineageParent{{Artifact: model, Relation: artifact.RelationTrainedFrom}},
	}, func(stage string) error {
		return os.WriteFile(filepath.Join(stage, trainingprogram.CheckpointWeights), []byte("trained weights"), 0o600)
	})
	testcheck.NoError(t, err)
	result := HeldoutTraining{Observation: derived(artifact.KindEvidence, "observation"), Model: model, Trained: checkpoint, Inputs: derived(artifact.KindProfile, "inputs")}
	if store == nil {
		return result
	}
	record, err := checkpoint.Batch("test-checkpoint/"+checkpoint.ID().String(), directory)
	testcheck.NoError(t, err)
	environment, err := runrecord.CurrentEnvironment("cpu", "host")
	testcheck.NoError(t, err)
	environmentContent, err := environment.Content()
	testcheck.NoError(t, err)
	recipeID := derived(artifact.KindRecipe, "recipe")
	record.Artifacts = append(record.Artifacts, artifact.Descriptor{ID: model}, artifact.Descriptor{ID: recipeID})
	record.Contents = append(record.Contents, environmentContent)
	if _, err := artifact.Publish(ctx, store, record); err != nil {
		t.Fatal(err)
	}
	observation, err := runrecord.PublishServingObservation(ctx, store, runrecord.ServingObservation{
		Model: model, Recipe: recipeID, Environment: environment.ID,
		Task: recipe.TaskTraining, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: 1,
	})
	testcheck.NoError(t, err)
	result.Observation = observation.ID
	return result
}

// score reports each held-out case's values shifted by offset and labeled
// label, as a model's forecasts would be.
func (f heldoutFixture) score(t *testing.T, offset float64, label string) HeldoutReport {
	t.Helper()
	var observations []HeldoutObservation
	for _, target := range f.cases {
		values := make([]float64, len(target.Values))
		for index, value := range target.Values {
			values[index] = value + offset
		}
		observations = append(observations, HeldoutObservation{Record: target.Record, Values: values, Label: label})
	}
	report, err := f.plan.Score(observations)
	testcheck.NoError(t, err)
	return report
}

// TestHeldoutVerdictRequiresStrictGainOnDeclaredMetric holds the verdict to
// the metric the objective declares, in that metric's direction: a trained
// model passes only by strictly improving it over the base model on the
// same held-out plan. An equal score, a metric no numeric scorer measures,
// reports scored for another objective's view, and a checkpoint that
// trained on another membership or model are refused or fail.
func TestHeldoutVerdictRequiresStrictGainOnDeclaredMetric(t *testing.T) {
	t.Parallel()
	forecast := newHeldoutFixture(t, trainingprogram.MetricForecastMAE)
	observation := heldoutTraining(t, nil, forecast.objective.Dataset, forecast.view.TrainingMembership)
	base, better, same := forecast.score(t, 2, "falling"), forecast.score(t, 1, "falling"), forecast.score(t, -2, "falling")
	for name, mismatched := range map[string]HeldoutTraining{
		"another membership": heldoutTraining(t, nil, forecast.objective.Dataset, forecast.view.HeldoutMembership),
		"another model":      {Observation: observation.Observation, Model: forecast.objective.Dataset, Trained: observation.Trained, Inputs: observation.Inputs},
	} {
		if _, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, better, mismatched); err == nil {
			t.Fatalf("a checkpoint that trained %s was judged", name)
		}
	}
	verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, better, observation)
	if err != nil || !verdict.Passed || verdict.BaselineValue != 2 || verdict.CandidateValue != 1 {
		t.Fatalf("a lower held-out MAE = (%+v, %v), want a passed verdict", verdict, err)
	}
	if verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, same, observation); err != nil || verdict.Passed {
		t.Fatalf("an equal held-out MAE = (passed=%t, %v), want a failed verdict", verdict.Passed, err)
	}
	if verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, better, base, observation); err != nil || verdict.Passed {
		t.Fatalf("a higher held-out MAE = (passed=%t, %v), want a failed verdict", verdict.Passed, err)
	}

	table := newHeldoutFixture(t, trainingprogram.MetricTableAccuracy)
	wrong, right := table.score(t, 0, "falling"), table.score(t, 0, "rising")
	if verdict, err := JudgeHeldout(table.objective, table.view, table.plan, wrong, right, observation); err != nil || !verdict.Passed {
		t.Fatalf("a higher held-out label accuracy = (passed=%t, %v), want a passed verdict", verdict.Passed, err)
	}
	if _, err := JudgeHeldout(forecast.objective, table.view, table.plan, wrong, right, observation); err == nil {
		t.Fatal("reports scored on another objective's view were judged")
	}

	psnr := forecast.objective
	psnr.Metric = trainingprogram.MetricImagePSNRUnit
	if _, err := JudgeHeldout(psnr, forecast.view, forecast.plan, base, better, observation); err == nil {
		t.Fatal("a metric no held-out numeric scorer measures was judged")
	}
}

// TestAVerdictNeedsAGainItsCasesSupport: one case's improvement is a strict gain and no supported one, a gain across the cases
// is, and a tie is neither, each reading the least gain its cases can tell from noise. Every plan keeps what the reading
// needs: a preference plan's loss and an unpaired plan's, record by record.
func TestAVerdictNeedsAGainItsCasesSupport(t *testing.T) {
	t.Parallel()
	forecast := newHeldoutFixture(t, trainingprogram.MetricForecastMAE)
	observation := heldoutTraining(t, nil, forecast.objective.Dataset, forecast.view.TrainingMembership)
	if len(forecast.cases) < minimumGainCases {
		t.Fatalf("the fixture holds %d cases", len(forecast.cases))
	}
	base := forecast.score(t, 2, "falling")
	shifted := func(offsets func(index int) float64) HeldoutReport {
		t.Helper()
		var observations []HeldoutObservation
		for index, target := range forecast.cases {
			values := slices.Clone(target.Values)
			for at := range values {
				values[at] += offsets(index)
			}
			observations = append(observations, HeldoutObservation{Record: target.Record, Values: values, Label: "falling"})
		}
		report, err := forecast.plan.Score(observations)
		testcheck.NoError(t, err)
		return report
	}
	for _, test := range []struct {
		name              string
		candidate         HeldoutReport
		passed, supported bool
	}{
		{"one case improves", shifted(func(index int) float64 {
			if index == 0 {
				return 1
			}
			return 2
		}), true, false},
		{"every case improves", shifted(func(index int) float64 { return 1 + 0.01*float64(index) }), true, true},
		{"every case improves alike", forecast.score(t, 1, "falling"), true, true},
		{"nothing moves", forecast.score(t, 2, "other"), false, false},
	} {
		verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, test.candidate, observation)
		if err != nil {
			t.Fatalf("%s: %v", test.name, err)
		}
		gain := verdict.ReadGain(base, test.candidate)
		if verdict.Passed != test.passed || gain.Supported != test.supported || gain.Cases != len(forecast.cases) {
			t.Errorf("%s: passed %t, gain %+v; want passed %t, supported %t", test.name, verdict.Passed, gain, test.passed, test.supported)
		}
	}
	lone := heldoutVerdict{Direction: runrecord.DirectionMaximize}
	if gain := lone.ReadGain(HeldoutReport{records: []RecordScore{{Value: 0}}}, HeldoutReport{records: []RecordScore{{Value: 1}}}); gain.Supported {
		t.Error("one record supported a gain")
	}
	if gain := lone.ReadGain(HeldoutReport{records: []RecordScore{{Record: "a"}, {Record: "b"}}}, HeldoutReport{records: []RecordScore{{Record: "c", Value: 1}, {Record: "d", Value: 1}}}); gain.Supported {
		t.Error("reports over different records supported a gain")
	}
	preference := preferenceTargetReport{Records: make([]preferenceRecordResult, 2)}
	preference.Records[1].Loss = 3
	if scores := preference.heldout().RecordScores(); len(scores) != 2 || scores[1].Value != 3 {
		t.Errorf("a preference report keeps %v", scores)
	}
	unpaired := unpairedTargetReport{Records: make([]unpairedRecordResult, 2)}
	if scores := unpaired.heldout([]float64{1, 2}).RecordScores(); len(scores) != 2 || scores[1].Value != 2 {
		t.Errorf("an unpaired report keeps %v", scores)
	}
}

// TestCommittedVerdictRescoresItsReports approves a committed forecast
// verdict only when each of its reports is what the cases it retains score
// to: a trained report that omits a held-out case, or one whose mean error
// was lowered without its cases, judges as passed but is refused once
// committed.
func TestCommittedVerdictRescoresItsReports(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Create(filepath.Join(t.TempDir(), "store"))
	testcheck.NoError(t, err)
	defer store.Close()
	forecast := newHeldoutFixture(t, trainingprogram.MetricForecastMAE)
	commit := heldoutCommitter(t, store)
	forecast.publishObjective(commit)
	session := heldoutTraining(t, store, forecast.objective.Dataset, forecast.view.TrainingMembership)
	spec := forecast.objective.ObjectiveSpec
	spec.Evidence = append(slices.Clone(spec.Evidence), session.Observation)
	current, err := trainingprogram.NewObjective(spec)
	testcheck.NoError(t, err)
	base, better := forecast.score(t, 2, "falling"), forecast.score(t, 1, "falling")
	commit("test-view", forecast.view.Batch, forecast.objective.Processors...)
	commit("test-plan", forecast.plan.Batch)
	commit("test-base", base.Batch)
	judge := func(key string, candidate HeldoutReport, wantPassed bool) (bool, error) {
		t.Helper()
		commit(key+"-report", candidate.Batch)
		verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, candidate, session)
		if err != nil || verdict.Passed != wantPassed {
			t.Fatalf("%s verdict = (passed=%t, %v), want passed=%t", key, verdict.Passed, err, wantPassed)
		}
		commit(key, verdict.Batch)
		passed, approvalErr := CommittedHeldoutVerdictPassed(ctx, store, verdict.ID, current)
		var visited []EvalSummary
		visitErr := VisitHeldoutVerdicts(ctx, store, func(_ artifact.ID, _, trained EvalSummary) error {
			visited = append(visited, trained)
			return nil
		}, nil)
		if approvalErr != nil {
			if visitErr == nil || len(visited) != 0 {
				t.Fatalf("%s invalid verdict reached reporting: %+v, %v", key, visited, visitErr)
			}
			// Told to go on, the visit reports the verdicts it cannot read and ends without error.
			var unreadable []artifact.ID
			if err := VisitHeldoutVerdicts(ctx, store, func(artifact.ID, EvalSummary, EvalSummary) error { return nil }, func(_, observation artifact.ID, _ error) {
				unreadable = append(unreadable, observation)
			}); err != nil || len(unreadable) == 0 {
				t.Fatalf("%s tolerant visit = (%v, %v), want the unreadable verdict reported and no error", key, unreadable, err)
			}
		} else if visitErr != nil || len(visited) == 0 || visited[0].Record != verdict.ID || visited[0].Metrics[string(verdict.Metric)] != verdict.CandidateValue {
			t.Fatalf("%s reporting = (%+v, %v), want the reconstructed value including a valid loss", key, visited, visitErr)
		}
		return passed, approvalErr
	}
	if passed, err := judge("test-honest", better, true); err != nil || !passed {
		t.Fatalf("the honest committed verdict = (%t, %v), want passed", passed, err)
	}
	if passed, err := judge("test-loss", forecast.score(t, 3, "falling"), false); err != nil || passed {
		t.Fatalf("the valid losing verdict = (%t, %v), want measured but not passed", passed, err)
	}
	retained, err := numericTargetReportCodec.Require(ctx, store, better.ID())
	testcheck.NoError(t, err)
	omitted := numericTargetReportCodec.Clone(retained)
	omitted.Records = omitted.Records[1:]
	omitted.Metrics[0].Value = 0
	for _, record := range omitted.Records {
		omitted.Metrics[0].Value += record.Scores[0].Value / float64(len(omitted.Records))
	}
	lowered := numericTargetReportCodec.Clone(retained)
	lowered.Metrics[0].Value /= 2
	for name, report := range map[string]numericTargetReport{"omitted": omitted, "lowered": lowered} {
		tampered, err := numericTargetReportCodec.New(report)
		testcheck.NoError(t, err)
		if passed, err := judge("test-"+name, tampered.heldout(), true); err == nil || passed || !strings.Contains(err.Error(), "does not rescore") {
			t.Fatalf("a committed verdict on the %s report = (%t, %v), want it refused", name, passed, err)
		}
	}
}
