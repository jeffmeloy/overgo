package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestRSICausalChainClosure pins the integrated chain: from a
// motivating observation through the controller proposal, the attempt,
// the experiment lifecycle, the promotion decision, serving evidence,
// and the rollback follow-up, every record carries the same causal
// root and the traversal reads typed causal fields only -- never
// chronology or aliases. Broken causal bindings refuse on every record
// family.
func TestRSICausalChainClosure(t *testing.T) {
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	motivation := id(artifact.KindEvidence, "motivating-observation")
	proposalEvidence := id(artifact.KindEvidence, "controller-proposal")
	root, err := NewCausalRoot(TriggerControllerProposal, proposalEvidence, motivation)
	if err != nil {
		t.Fatal(err)
	}

	attemptValue := fixtureAttempt(t)
	attemptValue.Causal = &root
	attempt, err := NewAttemptRecord(attemptValue)
	if err != nil {
		t.Fatal(err)
	}

	lifecycle := ExperimentLifecycle{
		Version: artifact.InitialDocumentVersion, State: ExperimentProposed,
		Experiment: id(artifact.KindEvidence, "experiment"),
		Evidence:   id(artifact.KindEvidence, "admission-decision"),
		Causal:     &root,
	}
	lifecycle, err = experimentLifecycleCodec.New(lifecycle)
	if err != nil {
		t.Fatal(err)
	}

	decision := ImprovementDecision{
		Version: artifact.SecondDocumentVersion, State: ImprovementPromote,
		Admission:   id(artifact.KindEvidence, "admission"),
		ParentModel: id(artifact.KindModel, "parent"), Incumbent: id(artifact.KindModel, "incumbent"),
		Candidate: id(artifact.KindModel, "candidate"), ChildModel: id(artifact.KindModel, "child"),
		Dataset:          id(artifact.KindDataset, "dataset"),
		DevelopmentSplit: id(artifact.KindDatasetShard, "dev"), PromotionSplit: id(artifact.KindDatasetShard, "promo"),
		Recipe: id(artifact.KindRecipe, "recipe"), Code: id(artifact.KindEvidence, "code"),
		Evaluator: id(artifact.KindEvidence, "evaluator"), Run: id(artifact.KindRun, "run"),
		Evaluation: id(artifact.KindEvaluation, "evaluation"), Proposer: id(artifact.KindEvidence, "proposer"),
		Authority: id(artifact.KindEvidence, "authority"), Decider: id(artifact.KindEvidence, "decider"),
		Rollback: id(artifact.KindModel, "parent"),
		Causal:   &root,
	}
	decision, err = improvementDecisionCodec.New(decision)
	if err != nil {
		t.Fatal(err)
	}

	servingValue := servingObservationFixture(t)
	servingValue.Version = artifact.InitialDocumentVersion
	servingValue.Causal = &root
	serving, err := servingObservationCodec.New(servingValue)
	if err != nil {
		t.Fatal(err)
	}

	// The rollback follow-up is caused by the failing serving evidence,
	// derived inside the same chain.
	rollbackCausal, err := root.Derive(TriggerRecovery, serving.ID)
	if err != nil {
		t.Fatal(err)
	}
	delivery := AutomationDeliveryAttempt{
		Version: artifact.InitialDocumentVersion,
		Plan:    id(artifact.KindProfile, "rollback-plan"), Operation: id(artifact.KindEvidence, "rollback-operation"),
		Tool: id(artifact.KindRecipe, "rollback-tool"), Destination: "serving-fleet",
		Outputs:     []artifact.ID{id(artifact.KindOutput, "rollback-notice")},
		Idempotency: id(artifact.KindEvidence, "rollback-idempotency"),
		State:       AutomationDeliveryAdmitted,
		Causal:      &rollbackCausal,
	}
	delivery, err = automationDeliveryAttemptCodec.New(delivery)
	if err != nil {
		t.Fatal(err)
	}
	rollbackCausal.Motivation[0] = id(artifact.KindEvidence, "mutated-caller-motivation")

	// The traversal: typed causal fields only. The rollback names the
	// serving evidence it recovers from; every record answers to the
	// proposal root; the root names its motivating observation.
	if delivery.Causal.RecoveredFrom != serving.ID {
		t.Fatalf("rollback does not name its cause: %+v", delivery.Causal)
	}
	for name, causal := range map[string]*CausalContext{
		"attempt": attempt.Causal, "lifecycle": lifecycle.Causal, "decision": decision.Causal,
		"serving": serving.Causal, "rollback": delivery.Causal,
	} {
		if causal == nil || causal.Root != proposalEvidence {
			t.Fatalf("%s left the causal chain: %+v", name, causal)
		}
		if len(causal.Motivation) != 1 || causal.Motivation[0] != motivation {
			t.Fatalf("%s lost the motivating observation: %+v", name, causal)
		}
	}

	// Every family refuses a causal binding whose fields disagree.
	broken := root
	broken.DelegatedFrom = id(artifact.KindEvidence, "not-my-trigger")
	attemptValue.Causal = &broken
	if _, err := NewAttemptRecord(attemptValue); err == nil {
		t.Fatal("attempt accepted a broken causal binding")
	}
	lifecycle.Causal, lifecycle.ID = &broken, artifact.ID{}
	if _, err := experimentLifecycleCodec.New(lifecycle); err == nil {
		t.Fatal("lifecycle accepted a broken causal binding")
	}
	decision.Causal, decision.ID = &broken, artifact.ID{}
	if _, err := improvementDecisionCodec.New(decision); err == nil {
		t.Fatal("decision accepted a broken causal binding")
	}
	servingValue.Causal = &broken
	if _, err := servingObservationCodec.New(servingValue); err == nil {
		t.Fatal("serving observation accepted a broken causal binding")
	}
	delivery.Causal, delivery.ID = &broken, artifact.ID{}
	if _, err := automationDeliveryAttemptCodec.New(delivery); err == nil {
		t.Fatal("delivery accepted a broken causal binding")
	}
}
