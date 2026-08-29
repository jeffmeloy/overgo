package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestScalingStudyRungsBindConfigsSeedsDataAndEvaluators(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	rung := func(ordinal uint32, name string) scalingStudyRung {
		return scalingStudyRung{
			Ordinal: ordinal, Config: id(artifact.KindRecipe, name+" config"), Model: id(artifact.KindModel, name+" model"),
			Seed: id(artifact.KindEvidence, name+" seed"), Dataset: id(artifact.KindDataset, name+" dataset"),
			Split: id(artifact.KindDatasetShard, name+" split"), DataOrder: id(artifact.KindEvidence, name+" order"),
			StartingCheckpoint: id(artifact.KindCheckpoint, name+" checkpoint"), Hardware: id(artifact.KindEvidence, name+" hardware"),
			Environment: id(artifact.KindEvidence, name+" environment"), Code: id(artifact.KindEvidence, name+" code"),
			Evaluator: id(artifact.KindProfile, name+" evaluator"), scalingRungFacts: scalingRungFacts{Planned: true},
		}
	}
	planned, failed, evaluated := rung(0, "small"), rung(1, "medium"), rung(2, "large")
	failed.Attempts = []scalingRungAttempt{{
		Run: id(artifact.KindRun, "failed run"), Outcome: OutcomeFailed, Terminal: id(artifact.KindEvidence, "failed terminal"),
	}}
	failed.Attempted, failed.Failed = true, true
	firstRun, retryRun := id(artifact.KindRun, "first large run"), id(artifact.KindRun, "retry large run")
	evaluated.Attempts = []scalingRungAttempt{
		{Run: firstRun, Outcome: OutcomeFailed, Terminal: id(artifact.KindEvidence, "first large terminal")},
		{Ordinal: 1, Run: retryRun, PriorRun: firstRun, Outcome: OutcomeSucceeded,
			Terminal: id(artifact.KindEvidence, "retry terminal"), Accounting: id(artifact.KindEvidence, "retry accounting")},
	}
	evaluated.EvaluationRun, evaluated.Evaluation = retryRun, id(artifact.KindEvidence, "large evaluation")
	evaluated.scalingRungFacts = scalingRungFacts{Planned: true, Attempted: true, Failed: true, Retried: true, Evaluated: true}
	study, err := scalingStudyCodec.New(scalingStudy{
		Version: artifact.InitialDocumentVersion, Protocol: id(artifact.KindProfile, "scaling protocol"),
		Rungs: []scalingStudyRung{evaluated, planned, failed},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := scalingStudyCodec.Content(study)
	if err != nil || content.Descriptor.ID != study.ID || study.Rungs[0].Ordinal != 0 || len(scalingStudyLineage(study)) < 35 {
		t.Fatalf("scaling study = (%+v, %v)", study, err)
	}
	hiddenFailure := study
	hiddenFailure.ID, hiddenFailure.Rungs = artifact.ID{}, cloneScalingStudyRungs(study.Rungs)
	hiddenFailure.Rungs[1].Failed = false
	if _, err := scalingStudyCodec.New(hiddenFailure); err == nil {
		t.Fatal("failed rung was hidden")
	}
	brokenRetry := study
	brokenRetry.ID, brokenRetry.Rungs = artifact.ID{}, cloneScalingStudyRungs(study.Rungs)
	brokenRetry.Rungs[2].Attempts[1].PriorRun = failed.Attempts[0].Run
	if _, err := scalingStudyCodec.New(brokenRetry); err == nil {
		t.Fatal("retry changed its prior run")
	}
	duplicate := study
	duplicate.ID, duplicate.Rungs = artifact.ID{}, cloneScalingStudyRungs(study.Rungs)
	duplicate.Rungs[1] = duplicate.Rungs[0]
	duplicate.Rungs[1].Ordinal = 1
	if _, err := scalingStudyCodec.New(duplicate); err == nil {
		t.Fatal("duplicate rung authorities admitted twice")
	}
}
