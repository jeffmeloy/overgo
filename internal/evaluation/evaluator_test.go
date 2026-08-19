package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestEvaluatorRanksKnownOutcomesUnderPriorAuthority(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	acceptance, err := newAcceptancePolicy([]runrecord.Metric{{Name: "quality", Direction: runrecord.DirectionMaximize}})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := NewEvaluator(id(artifact.KindProfile, "plan"), acceptance)
	if err != nil {
		t.Fatal(err)
	}
	proposer := runrecord.AuthorityDomain{Name: "proposer", Identity: id(artifact.KindEvidence, "proposer")}
	decider := runrecord.AuthorityDomain{Name: "decider", Identity: id(artifact.KindEvidence, "decider")}
	metric := func(value float64) []runrecord.Metric {
		return []runrecord.Metric{{Name: "quality", Value: value, Direction: runrecord.DirectionMaximize}}
	}
	outcomes := []KnownOutcome{
		{Model: id(artifact.KindModel, "known good"), Rank: 0, Metrics: metric(2)},
		{Model: id(artifact.KindModel, "known bad"), Rank: 1, Metrics: metric(1)},
	}
	outcomeSet, err := artifact.JSONID(artifact.KindDatasetShard, outcomes)
	if err != nil {
		t.Fatal(err)
	}
	prior, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Generation: 0, Proposer: proposer,
		Evaluator: runrecord.AuthorityDomain{Name: "evaluator-v1", Identity: id(artifact.KindEvidence, "incumbent")},
		Decider:   decider, SealedInputs: outcomeSet,
		CleanWorker: id(artifact.KindEvidence, "clean worker"),
	})
	if err != nil {
		t.Fatal(err)
	}
	approval, err := recipe.NewDecision(candidate.ID, recipe.DecisionAccepted, recipe.EvidenceProduction, "",
		recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: decider.Identity},
		[]artifact.ID{prior.SealedInputs})
	if err != nil {
		t.Fatal(err)
	}
	current, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Generation: 1, Proposer: proposer,
		Evaluator: runrecord.AuthorityDomain{Name: "evaluator-v2", Identity: candidate.ID},
		Decider:   decider, SealedInputs: prior.SealedInputs, CleanWorker: prior.CleanWorker, PriorApproval: approval.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateEvaluatorPromotion(candidate, outcomeSet, outcomes, current, prior, approval); err != nil {
		t.Fatal(err)
	}
	outcomes[0].Metrics, outcomes[1].Metrics = outcomes[1].Metrics, outcomes[0].Metrics
	if err := ValidateEvaluatorPromotion(candidate, outcomeSet, outcomes, current, prior, approval); err == nil {
		t.Fatal("misranking evaluator promoted")
	}
}
