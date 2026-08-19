package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestTextTargetModalitiesUseDeclaredScorers(t *testing.T) {
	verifier := testutil.ArtifactID(t, artifact.KindProfile, "text target verifier")
	likelihood := -1.25
	for _, modality := range []recipecontract.Modality{
		recipecontract.ModalityText, recipecontract.ModalityImage,
		recipecontract.ModalityAudio, recipecontract.ModalityVideo,
	} {
		view := textTargetView(t, modality)
		suite := TextTargetSuite{
			Scorers: []TextScorerSpec{
				{Kind: TextContinuationLikelihood},
				{Kind: TextGeneratedAnswer, Normalization: []TextNormalization{TextTrimSpaceNormalization, TextLowercaseNormalization}},
				{Kind: TextStructuredValidity, Verifier: artifact.IDPointer(verifier)},
				{Kind: TextOCRFields, Normalization: []TextNormalization{TextTrimSpaceNormalization}, Verifier: artifact.IDPointer(verifier)},
				{Kind: TextVQARules, Verifier: artifact.IDPointer(verifier)},
			},
			Cases: []TextTargetCase{{
				Record: "heldout", Answers: []string{"yes"},
				Fields: []NamedText{{Name: "title", Value: "value"}}, Rules: []string{"supported"},
			}},
		}
		plan, err := CompileTextTargetPlan(view, suite)
		if err != nil {
			t.Fatalf("%s: %v", modality, err)
		}
		report, err := ScoreTextTargets(plan, []TextTargetObservation{{
			Record: "heldout", Raw: " YES ", ContinuationLogLikelihood: &likelihood,
			Verifiers: []TextVerifierResult{
				{Verifier: verifier, Passed: true},
				{Verifier: verifier, Name: "title", Value: "value"},
				{Verifier: verifier, Name: "supported", Passed: true},
			},
		}})
		if err != nil || len(report.Metrics) != len(suite.Scorers) || report.Observations[0].Raw != " YES " {
			t.Fatalf("%s report = %+v, err=%v", modality, report, err)
		}
	}
	view := textTargetView(t, recipecontract.ModalityText)
	plan, err := CompileTextTargetPlan(view, TextTargetSuite{
		Scorers: []TextScorerSpec{{Kind: TextGeneratedAnswer}},
		Cases:   []TextTargetCase{{Record: "heldout", Answers: []string{"answer"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := ScoreTextTargets(plan, []TextTargetObservation{{Record: "heldout", Raw: "answer", ContinuationLogLikelihood: &likelihood}})
	if err != nil || len(report.Metrics) != 1 || report.Metrics[0].Name != string(TextGeneratedAnswer) {
		t.Fatalf("undeclared scorer affected report: %+v, err=%v", report, err)
	}
}

func textTargetView(t *testing.T, input recipecontract.Modality) SFTEvaluationView {
	t.Helper()
	view, err := sftEvaluationViewCodec.New(SFTEvaluationView{
		Version:            sftEvaluationViewVersion,
		Objective:          testutil.ArtifactID(t, artifact.KindProfile, "text target objective"),
		Dataset:            testutil.ArtifactID(t, artifact.KindDataset, "text target dataset"),
		TrainingMembership: testutil.ArtifactID(t, artifact.KindDatasetShard, "text target training"),
		HeldoutMembership:  testutil.ArtifactID(t, artifact.KindDatasetShard, "text target heldout"),
		Processors:         []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "text target processor")},
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{input}, Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Records: []dataset.Record{{ID: "heldout", Group: "heldout"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return view
}
