package evaluation

import (
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
)

// heldoutFixture is a forecast objective bound to the training half of a
// real group split, its held-out view, and a numeric plan over the held-out
// records.
type heldoutFixture struct {
	objective trainingprogram.ObjectiveDocument
	view      SFTEvaluationView
	plan      NumericTargetPlan
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
	suite := NumericTargetSuite{Scorers: []NumericScorerSpec{{Kind: NumericAbsoluteError}, {Kind: NumericLabelAccuracy}}}
	for index, record := range view.Records {
		suite.Cases = append(suite.Cases, NumericTargetCase{Record: record.ID, Values: []float64{float64(index), 1}, Label: "rising"})
	}
	plan, err := CompileNumericTargetPlan(view, suite)
	if err != nil {
		t.Fatal(err)
	}
	return heldoutFixture{objective: objective, view: view, plan: plan, split: split}
}

// score reports each held-out case's values shifted by offset and labeled
// label, as a model's forecasts would be.
func (f heldoutFixture) score(t *testing.T, offset float64, label string) NumericTargetReport {
	t.Helper()
	var observations []NumericTargetObservation
	for _, target := range f.plan.Cases {
		values := make([]float64, len(target.Values))
		for index, value := range target.Values {
			values[index] = value + offset
		}
		observations = append(observations, NumericTargetObservation{Record: target.Record, Values: values, Label: label})
	}
	report, err := ScoreNumericTargets(f.plan, observations)
	if err != nil {
		t.Fatal(err)
	}
	return report
}

// TestHeldoutVerdictRequiresStrictGainOnDeclaredMetric holds the verdict to
// the metric the objective declares, in that metric's direction: a trained
// model passes only by strictly improving it over the base model on the
// same held-out plan. An equal score, a metric no numeric scorer measures,
// and reports scored for another objective's view are refused or fail.
func TestHeldoutVerdictRequiresStrictGainOnDeclaredMetric(t *testing.T) {
	t.Parallel()
	forecast := newHeldoutFixture(t, trainingprogram.MetricForecastMAE)
	observation := forecast.objective.Evidence[0]
	base, better, same := forecast.score(t, 2, "falling"), forecast.score(t, 1, "falling"), forecast.score(t, -2, "falling")
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
