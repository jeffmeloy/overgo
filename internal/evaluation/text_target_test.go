package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestTextTargetModalitiesUseDeclaredScorers(t *testing.T) {
	t.Parallel()
	verifier := testutil.ArtifactID(t, artifact.KindProfile, "text target verifier")
	likelihood := -1.25
	for _, modality := range []recipecontract.Modality{
		recipecontract.ModalityText, recipecontract.ModalityImage,
		recipecontract.ModalityAudio, recipecontract.ModalityVideo,
	} {
		view := textTargetView(t, modality)
		suite := textTargetSuite{
			Scorers: []textScorerSpec{
				{Kind: textContinuationLikelihood},
				{Kind: textGeneratedAnswer, Normalization: []textNormalization{textTrimSpaceNormalization, textLowercaseNormalization}},
				{Kind: textStructuredValidity, Verifier: artifact.IDPointer(verifier)},
				{Kind: textOCRFields, Normalization: []textNormalization{textTrimSpaceNormalization}, Verifier: artifact.IDPointer(verifier)},
				{Kind: textVQARules, Verifier: artifact.IDPointer(verifier)},
			},
			Cases: []textTargetCase{{
				Record: "heldout", Answers: []string{"yes"},
				Fields: []namedText{{Name: "title", Value: "value"}}, Rules: []string{"supported"},
			}},
		}
		plan, err := compileTextTargetPlan(view, suite)
		if err != nil {
			t.Fatalf("%s: %v", modality, err)
		}
		report, err := scoreTextTargets(plan, []textTargetObservation{{
			Record: "heldout", Raw: " YES ", ContinuationLogLikelihood: &likelihood,
			Verifiers: []textVerifierResult{
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
	plan, err := compileTextTargetPlan(view, textTargetSuite{
		Scorers: []textScorerSpec{{Kind: textGeneratedAnswer}},
		Cases:   []textTargetCase{{Record: "heldout", Answers: []string{"answer"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := scoreTextTargets(plan, []textTargetObservation{{Record: "heldout", Raw: "answer", ContinuationLogLikelihood: &likelihood}})
	if err != nil || len(report.Metrics) != 1 || report.Metrics[0].Name != string(textGeneratedAnswer) {
		t.Fatalf("undeclared scorer affected report: %+v, err=%v", report, err)
	}
}

func textTargetView(t *testing.T, input recipecontract.Modality) SFTEvaluationView {
	t.Helper()
	view, err := sftEvaluationViewCodec.New(SFTEvaluationView{
		Version:            artifact.InitialDocumentVersion,
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
