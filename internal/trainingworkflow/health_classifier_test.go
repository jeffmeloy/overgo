package trainingworkflow

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestTrainingHealthClassifiersArePhaseAwareRobustAndReopenable(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	phase, mixture, boundary := id(artifact.KindProfile, "train phase"), id(artifact.KindDatasetShard, "mixture"), id(artifact.KindEvidence, "boundary")
	fact := func(kind healthClassifierKind, name string) healthClassifierFact {
		return healthClassifierFact{
			Kind: kind, Signal: id(artifact.KindProfile, name+" signal"), Phase: phase, Mixture: mixture,
			Observation: id(artifact.KindEvidence, name+" observation"), Baseline: id(artifact.KindEvidence, name+" baseline"),
			Method: id(artifact.KindProfile, name+" method"), WindowEvidence: id(artifact.KindEvidence, name+" window"), WindowAdmitted: true,
			PersistenceEvidence: id(artifact.KindEvidence, name+" persistence"), PersistenceAdmitted: true,
			BoundaryEvidence: boundary, ReopenTrigger: id(artifact.KindRecipe, name+" reopen"), Triggered: true,
		}
	}
	absent := fact(healthClassifierAbsentTelemetry, "absent")
	stall := fact(healthClassifierStall, "stall")
	stall.Observed = true
	nonfinite := fact(healthClassifierNonfinite, "nonfinite")
	nonfinite.Observed = true
	regression := fact(healthClassifierSustainedRegression, "regression")
	regression.Observed, regression.Finite, regression.RobustRegression = true, true, true
	classification, err := healthClassificationCodec.New(healthClassification{
		Version: artifact.InitialDocumentVersion, Run: id(artifact.KindRun, "run"), HealthEvidence: id(artifact.KindEvidence, "health"),
		Policy: id(artifact.KindProfile, "classifier policy"), Facts: []healthClassifierFact{regression, nonfinite, absent, stall},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := healthClassificationCodec.Content(classification)
	if err != nil || content.Descriptor.ID != classification.ID || len(healthClassificationLineage(classification)) != 34 {
		t.Fatalf("health classification = (%+v, %v)", classification, err)
	}
	healthyZero := classification
	healthyZero.ID, healthyZero.Facts = artifact.ID{}, slices.Clone(classification.Facts)
	for index := range healthyZero.Facts {
		if healthyZero.Facts[index].Kind == healthClassifierAbsentTelemetry {
			healthyZero.Facts[index].Observed, healthyZero.Facts[index].KnownZero = true, true
			healthyZero.Facts[index].Finite, healthyZero.Facts[index].Triggered = true, false
		}
	}
	if _, err := healthClassificationCodec.New(healthyZero); err != nil {
		t.Fatalf("healthy observed zero was classified as absent: %v", err)
	}
	mixedPhase := classification
	mixedPhase.ID, mixedPhase.Facts = artifact.ID{}, slices.Clone(classification.Facts)
	mixedPhase.Facts[0].Phase = id(artifact.KindProfile, "next phase")
	if _, err := healthClassificationCodec.New(mixedPhase); err == nil {
		t.Fatal("phase boundary was mixed into one classifier window")
	}
	notRobust := classification
	notRobust.ID, notRobust.Facts = artifact.ID{}, slices.Clone(classification.Facts)
	for index := range notRobust.Facts {
		if notRobust.Facts[index].Kind == healthClassifierSustainedRegression {
			notRobust.Facts[index].RobustRegression = false
		}
	}
	if _, err := healthClassificationCodec.New(notRobust); err == nil {
		t.Fatal("unsustained robust verdict was retained as triggered")
	}
	missingReopen := classification
	missingReopen.ID, missingReopen.Facts = artifact.ID{}, slices.Clone(classification.Facts)
	missingReopen.Facts[0].ReopenTrigger = artifact.ID{}
	if _, err := healthClassificationCodec.New(missingReopen); err == nil {
		t.Fatal("classifier without a reopen trigger admitted")
	}
}
