package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestMediaTargetMetricsRequireExplicitVerifierAuthority(t *testing.T) {
	t.Parallel()
	verifier := testutil.ArtifactID(t, artifact.KindProfile, "media verifier")
	for _, modality := range []recipecontract.Modality{
		recipecontract.ModalityAudio, recipecontract.ModalityImage, recipecontract.ModalityVideo,
	} {
		view := textTargetView(t, recipecontract.ModalityText)
		view.Signature.Outputs[0] = modality
		view, err := sftEvaluationViewCodec.New(view)
		if err != nil {
			t.Fatal(err)
		}
		expected := testutil.ArtifactID(t, artifact.KindOutput, "expected "+string(modality))
		output := testutil.ArtifactID(t, artifact.KindOutput, "output "+string(modality))
		plan, err := compileMediaTargetPlan(view, mediaTargetSuite{
			Oracles: []mediaOracle{mediaDecode, mediaArtifactIdentity},
			Scorers: []mediaScorerSpec{{
				Name: "declared-score", Kind: mediaIndependentScorer, Verifier: verifier,
				Direction: runrecord.DirectionMaximize,
			}},
			Cases: []mediaTargetCase{{Record: "heldout", Expected: expected}},
		})
		if err != nil {
			t.Fatal(err)
		}
		observation := mediaTargetObservation{
			Record: "heldout", Output: output,
			Oracles:   []mediaOracleResult{{Kind: mediaArtifactIdentity, Passed: true}, {Kind: mediaDecode, Passed: true}},
			Verifiers: []mediaVerifierResult{{Name: "declared-score", Verifier: verifier, Value: 0.75}},
		}
		report, err := scoreMediaTargets(plan, []mediaTargetObservation{observation})
		if err != nil || len(report.Metrics) != 1 || report.Metrics[0].Value != observation.Verifiers[0].Value {
			t.Fatalf("%s report=%+v err=%v", modality, report, err)
		}
		observation.Verifiers[0].Verifier = testutil.ArtifactID(t, artifact.KindProfile, "undeclared verifier")
		if _, err := scoreMediaTargets(plan, []mediaTargetObservation{observation}); err == nil {
			t.Fatalf("%s accepted undeclared verifier", modality)
		}
	}
}
