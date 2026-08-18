package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestMediaTargetMetricsRequireExplicitVerifierAuthority(t *testing.T) {
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
		plan, err := CompileMediaTargetPlan(view, MediaTargetSuite{
			Oracles: []MediaOracle{MediaDecode, MediaArtifactIdentity},
			Scorers: []MediaScorerSpec{{
				Name: "declared-score", Kind: MediaIndependentScorer, Verifier: verifier,
				Direction: runrecord.DirectionMaximize,
			}},
			Cases: []MediaTargetCase{{Record: "heldout", Expected: expected}},
		})
		if err != nil {
			t.Fatal(err)
		}
		observation := MediaTargetObservation{
			Record: "heldout", Output: output,
			Oracles:   []MediaOracleResult{{Kind: MediaArtifactIdentity, Passed: true}, {Kind: MediaDecode, Passed: true}},
			Verifiers: []MediaVerifierResult{{Name: "declared-score", Verifier: verifier, Value: 0.75}},
		}
		report, err := ScoreMediaTargets(plan, []MediaTargetObservation{observation})
		if err != nil || len(report.Metrics) != 1 || report.Metrics[0].Value != observation.Verifiers[0].Value {
			t.Fatalf("%s report=%+v err=%v", modality, report, err)
		}
		observation.Verifiers[0].Verifier = testutil.ArtifactID(t, artifact.KindProfile, "undeclared verifier")
		if _, err := ScoreMediaTargets(plan, []MediaTargetObservation{observation}); err == nil {
			t.Fatalf("%s accepted undeclared verifier", modality)
		}
	}
}
