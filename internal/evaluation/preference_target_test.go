package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestPreferenceEvaluationBindsFrozenReferenceAndHeldoutSplit(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	datasetID := id(artifact.KindDataset, "preference dataset")
	trainingID := id(artifact.KindDatasetShard, "preference training")
	heldoutID := id(artifact.KindDatasetShard, "preference heldout")
	signature := recipecontract.ModalitySignature{
		Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText},
	}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "preference", Kind: trainingprogram.ObjectiveDPO, Signature: signature,
		Dataset: datasetID, Split: trainingID, Processors: []artifact.ID{id(artifact.KindProfile, "preference processor")},
		Loss: id(artifact.KindProfile, "preference loss"), Evaluation: id(artifact.KindProfile, "preference evaluation"),
		Metric: trainingprogram.MetricTokenAccuracy, Evidence: []artifact.ID{id(artifact.KindEvidence, "preference evidence")},
		Authority: trainingprogram.ObjectiveApproved,
	})
	if err != nil {
		t.Fatal(err)
	}
	view, err := sftEvaluationViewCodec.New(SFTEvaluationView{
		Version: sftEvaluationViewVersion, Objective: objective.ID, Dataset: datasetID,
		TrainingMembership: trainingID, HeldoutMembership: heldoutID,
		Processors: objective.Processors, Signature: signature,
		Records: []dataset.Record{{ID: "pair", Group: "heldout-group"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	policy := trainingprogram.PreferencePolicy{Reference: id(artifact.KindModel, "frozen reference"), Scale: 0.2}
	plan, err := CompilePreferenceTargetPlan(objective, view, PreferenceTargetSuite{Policy: policy})
	if err != nil || plan.Policy.Reference != policy.Reference || plan.Heldout != heldoutID {
		t.Fatalf("plan=%+v err=%v", plan, err)
	}
	score := func(value float64) sequencescore.Score { return sequencescore.Score{LogProbability: value, Tokens: 1} }
	report, err := ScorePreferenceTargets(plan, []PreferenceTargetObservation{{
		Record: "pair", PolicyChosen: score(-1), PolicyRejected: score(-3),
		ReferenceChosen: score(-2), ReferenceRejected: score(-3),
	}})
	if err != nil || len(report.Metrics) != 3 || !report.Records[0].Chosen || report.Records[0].RelativeMargin != 1 {
		t.Fatalf("report=%+v err=%v", report, err)
	}
}
