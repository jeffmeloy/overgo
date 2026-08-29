package trainingworkflow

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestHealthEvidenceCannotMutateRunsOutsideLifecycleAuthority(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	phase, mixture, boundary := id(artifact.KindProfile, "phase"), id(artifact.KindDatasetShard, "mixture"), id(artifact.KindEvidence, "boundary")
	facts := make([]healthClassifierFact, len(healthClassifierKinds))
	for index, kind := range healthClassifierKinds {
		facts[index] = healthClassifierFact{
			Kind: kind, Signal: id(artifact.KindProfile, string(kind)+" signal"), Phase: phase, Mixture: mixture,
			Observation: id(artifact.KindEvidence, string(kind)+" observation"), Baseline: id(artifact.KindEvidence, string(kind)+" baseline"),
			Method: id(artifact.KindProfile, string(kind)+" method"), WindowEvidence: id(artifact.KindEvidence, string(kind)+" window"),
			PersistenceEvidence: id(artifact.KindEvidence, string(kind)+" persistence"), BoundaryEvidence: boundary,
			ReopenTrigger: id(artifact.KindRecipe, string(kind)+" reopen"),
		}
	}
	facts[1].Observed, facts[1].Advanced = true, true
	facts[2].Observed, facts[2].Finite = true, true
	facts[3].Observed, facts[3].Finite = true, true
	classification, err := healthClassificationCodec.New(healthClassification{
		Version: artifact.InitialDocumentVersion, Run: id(artifact.KindRun, "run"),
		HealthEvidence: id(artifact.KindEvidence, "health"), Policy: id(artifact.KindProfile, "policy"), Facts: facts,
	})
	if err != nil {
		t.Fatal(err)
	}
	advisory, err := buildHealthOperatorDecision(classification, healthOperatorAdvisory, nil)
	if err != nil || !advisory.AdvisoryOnly || len(healthOperatorDecisionLineage(advisory)) != 4 {
		t.Fatalf("health advisory = (%+v, %v)", advisory, err)
	}
	if _, err := buildHealthOperatorDecision(classification, healthOperatorContainment, nil); err == nil {
		t.Fatal("health classification mutated a run without lifecycle authority")
	}
	experiment := id(artifact.KindEvidence, "experiment")
	proposed, err := runrecord.NewExperimentLifecycle(runrecord.ExperimentLifecycle{
		State: runrecord.ExperimentProposed, Experiment: experiment, Evidence: id(artifact.KindEvidence, "proposal"),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	admitted, err := runrecord.NewExperimentLifecycle(runrecord.ExperimentLifecycle{
		State: runrecord.ExperimentAdmitted, Experiment: experiment, Evidence: id(artifact.KindEvidence, "admission"),
	}, &proposed)
	if err != nil {
		t.Fatal(err)
	}
	contained, err := runrecord.NewExperimentLifecycle(runrecord.ExperimentLifecycle{
		State: runrecord.ExperimentContained, Experiment: experiment, Evidence: classification.ID,
	}, &admitted)
	if err != nil {
		t.Fatal(err)
	}
	authorized, err := buildHealthOperatorDecision(classification, healthOperatorContainment, &contained)
	if err != nil || authorized.AdvisoryOnly || authorized.Lifecycle != contained.ID || len(healthOperatorDecisionLineage(authorized)) != 6 {
		t.Fatalf("authorized containment = (%+v, %v)", authorized, err)
	}
	foreign, err := runrecord.NewExperimentLifecycle(runrecord.ExperimentLifecycle{
		State: runrecord.ExperimentContained, Experiment: experiment, Evidence: id(artifact.KindEvidence, "foreign classification"),
	}, &admitted)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := buildHealthOperatorDecision(classification, healthOperatorContainment, &foreign); err == nil {
		t.Fatal("foreign lifecycle evidence authorized health mutation")
	}
}
