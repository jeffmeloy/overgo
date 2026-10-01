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

func newHeldoutFixture(t *testing.T, metric trainingprogram.EvaluationMetric) heldoutFixture {
	t.Helper()
	derived := func(kind artifact.Kind, role string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte("overgo/heldout-verdict-test/"+role))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	source := derived(artifact.KindDataset, "dataset")
	var records []dataset.Record
	for index := range 12 {
		records = append(records, dataset.Record{ID: fmt.Sprintf("series/%02d", index), Group: fmt.Sprintf("series-%02d", index)})
	}
	split, err := dataset.BuildGroupSplit(source, records, 17, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	view, err := CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		t.Fatal(err)
	}
	var cases []HeldoutCase
	for index, record := range view.Records {
		cases = append(cases, HeldoutCase{Record: record.ID, Values: []float64{float64(index), 1}, Label: "rising"})
	}
	plan, err := CompileHeldoutPlan(objective, view, cases)
	if err != nil {
		t.Fatal(err)
	}
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
		id, err := artifact.IdentifyBytes(kind, []byte("overgo/heldout-training-test/"+role))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	const parameters = 4
	plan, err := hostoptimizer.CompilePlan(parameters, []hostoptimizer.GroupSpec{{Name: "weight", End: parameters, Rows: 2, Cols: 2}})
	if err != nil {
		t.Fatal(err)
	}
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(parameters)
	if err != nil {
		t.Fatal(err)
	}
	muon, err := hostoptimizer.New(make([]float32, parameters), slices.Repeat([]float32{1}, parameters), plan, config)
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	result := HeldoutTraining{Observation: derived(artifact.KindEvidence, "observation"), Model: model, Trained: checkpoint, Inputs: derived(artifact.KindProfile, "inputs")}
	if store == nil {
		return result
	}
	record, err := checkpoint.Batch("test-checkpoint/"+checkpoint.ID().String(), directory)
	if err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.CurrentEnvironment("cpu", "host")
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
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

// TestCommittedVerdictRescoresItsReports approves a committed forecast
// verdict only when each of its reports is what the cases it retains score
// to: a trained report that omits a held-out case, or one whose mean error
// was lowered without its cases, judges as passed but is refused once
// committed.
func TestCommittedVerdictRescoresItsReports(t *testing.T) {
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
	spec := forecast.objective.ObjectiveSpec
	spec.Evidence = append(slices.Clone(spec.Evidence), session.Observation)
	current, err := trainingprogram.NewObjective(spec)
	if err != nil {
		t.Fatal(err)
	}
	base, better := forecast.score(t, 2, "falling"), forecast.score(t, 1, "falling")
	commit("test-view", forecast.view.Batch, forecast.objective.Processors...)
	commit("test-plan", forecast.plan.Batch)
	commit("test-base", base.Batch)
	judge := func(key string, candidate HeldoutReport) (bool, error) {
		t.Helper()
		commit(key+"-report", candidate.Batch)
		verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, candidate, session)
		if err != nil || !verdict.Passed {
			t.Fatalf("%s verdict = (passed=%t, %v), want a passed judgment", key, verdict.Passed, err)
		}
		commit(key, verdict.Batch)
		return CommittedHeldoutVerdictPassed(ctx, store, verdict.ID, current)
	}
	if passed, err := judge("test-honest", better); err != nil || !passed {
		t.Fatalf("the honest committed verdict = (%t, %v), want passed", passed, err)
	}
	retained, err := numericTargetReportCodec.Require(ctx, store, better.ID())
	if err != nil {
		t.Fatal(err)
	}
	omitted := cloneNumericTargetReport(retained)
	omitted.Records = omitted.Records[1:]
	omitted.Metrics[0].Value = 0
	for _, record := range omitted.Records {
		omitted.Metrics[0].Value += record.Scores[0].Value / float64(len(omitted.Records))
	}
	lowered := cloneNumericTargetReport(retained)
	lowered.Metrics[0].Value /= 2
	for name, report := range map[string]numericTargetReport{"omitted": omitted, "lowered": lowered} {
		tampered, err := numericTargetReportCodec.New(report)
		if err != nil {
			t.Fatal(err)
		}
		if passed, err := judge("test-"+name, tampered.heldout()); err == nil || passed || !strings.Contains(err.Error(), "does not rescore") {
			t.Fatalf("a committed verdict on the %s report = (%t, %v), want it refused", name, passed, err)
		}
	}
}
