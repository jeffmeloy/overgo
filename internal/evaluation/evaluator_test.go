package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestEvaluatorRanksKnownOutcomesUnderPriorAuthority(t *testing.T) {
	t.Parallel()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	acceptance := testcheck.Must(newAcceptancePolicy([]runrecord.Metric{{Name: "quality", Direction: runrecord.DirectionMaximize}}))(t)
	candidate := testcheck.Must(NewEvaluator(id(artifact.KindProfile, "plan"), acceptance))(t)
	proposer := runrecord.AuthorityDomain{Name: "proposer", Identity: id(artifact.KindEvidence, "proposer")}
	decider := runrecord.AuthorityDomain{Name: "decider", Identity: id(artifact.KindEvidence, "decider")}
	metric := func(value float64) []runrecord.Metric {
		return []runrecord.Metric{{Name: "quality", Value: value, Direction: runrecord.DirectionMaximize}}
	}
	outcomes := []KnownOutcome{
		{Model: id(artifact.KindModel, "known good"), Rank: 0, Metrics: metric(2)},
		{Model: id(artifact.KindModel, "known bad"), Rank: 1, Metrics: metric(1)},
	}
	outcomeSet := testcheck.Must(artifact.JSONID(artifact.KindDatasetShard, outcomes))(t)
	prior := testcheck.Must(runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Generation: 0, Proposer: proposer,
		Evaluator: runrecord.AuthorityDomain{Name: "evaluator-v1", Identity: id(artifact.KindEvidence, "incumbent")},
		Decider:   decider, SealedInputs: outcomeSet,
		CleanWorker: id(artifact.KindEvidence, "clean worker"),
	}))(t)
	approval := testcheck.Must(recipe.NewDecision(candidate.ID, recipe.DecisionAccepted, recipe.EvidenceProduction, "",
		recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: decider.Identity},
		[]artifact.ID{prior.SealedInputs}))(t)
	current := testcheck.Must(runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Generation: 1, Proposer: proposer,
		Evaluator: runrecord.AuthorityDomain{Name: "evaluator-v2", Identity: candidate.ID},
		Decider:   decider, SealedInputs: prior.SealedInputs, CleanWorker: prior.CleanWorker, PriorApproval: approval.ID,
	}))(t)
	testcheck.NoError(t, ValidateEvaluatorPromotion(candidate, outcomeSet, outcomes, current, prior, approval))
	outcomes[0].Metrics, outcomes[1].Metrics = outcomes[1].Metrics, outcomes[0].Metrics
	if err := ValidateEvaluatorPromotion(candidate, outcomeSet, outcomes, current, prior, approval); err == nil {
		t.Fatal("misranking evaluator promoted")
	}
}
