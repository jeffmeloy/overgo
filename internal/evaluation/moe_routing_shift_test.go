package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMoERoutingEvaluationDetectsNonIIDMixtureAndLagFailures(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	segment := func(name string, admitted bool, source artifact.ID) moeRoutingShiftSegment {
		result := moeRoutingShiftSegment{
			Stratum: id(artifact.KindDatasetShard, name), Evaluation: id(artifact.KindEvidence, name+"-evaluation"),
			Admitted: admitted,
		}
		if source.Valid() {
			result.BiasSource = artifact.IDPointer(source)
		}
		return result
	}
	firstTransition := segment("transition-a", true, artifact.ID{})
	secondTransition := segment("transition-b", false, artifact.ID{})
	firstTransition.BiasSource = artifact.IDPointer(firstTransition.Stratum)
	secondTransition.BiasSource = artifact.IDPointer(firstTransition.Stratum)
	cases := []moeRoutingShiftCase{
		{Kind: moeRoutingTransition, AggregateEvaluation: id(artifact.KindEvidence, "transition"), Segments: []moeRoutingShiftSegment{firstTransition, secondTransition}},
		{Kind: moeRoutingMixed, AggregateEvaluation: id(artifact.KindEvidence, "mixed"), AggregateAdmitted: true, Segments: []moeRoutingShiftSegment{
			segment("mixed-a", true, artifact.ID{}), segment("mixed-b", false, artifact.ID{}),
		}},
		{Kind: moeRoutingHomogeneous, AggregateEvaluation: id(artifact.KindEvidence, "homogeneous"), AggregateAdmitted: true, Segments: []moeRoutingShiftSegment{
			segment("homogeneous-a", true, artifact.ID{}),
		}},
		{Kind: moeRoutingImbalanced, AggregateEvaluation: id(artifact.KindEvidence, "imbalanced"), Segments: []moeRoutingShiftSegment{
			segment("imbalanced-a", false, artifact.ID{}), segment("imbalanced-b", false, artifact.ID{}),
		}},
	}
	evidence, err := moeRoutingShiftCodec.New(moeRoutingShiftEvidence{
		Version: artifact.InitialDocumentVersion, BaselineMatrix: id(artifact.KindRecipe, "baseline-matrix"),
		Cases: cases, HiddenStratumFailure: true, ControllerLag: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := moeRoutingShiftCodec.Content(evidence)
	if err != nil || content.Descriptor.ID != evidence.ID || len(moeRoutingShiftLineage(evidence)) < 10 {
		t.Fatalf("shift evidence = (%+v, %v)", evidence, err)
	}
	wrong := evidence
	wrong.ID, wrong.ControllerLag = artifact.ID{}, false
	if _, err := moeRoutingShiftCodec.New(wrong); err == nil {
		t.Fatal("transition lag hidden by stored verdict")
	}
	missing := evidence
	missing.ID, missing.Cases = artifact.ID{}, missing.Cases[:3]
	if _, err := moeRoutingShiftCodec.New(missing); err == nil {
		t.Fatal("missing shift case admitted")
	}
}
