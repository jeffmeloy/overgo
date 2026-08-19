package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestEvaluatorRanksKnownOutcomesUnderPriorAuthority pins the evaluator
// -as-artifact contract: a candidate evaluator promotes only when it ranks
// every known-good sealed-oracle model strictly better than every known-bad
// one AND the promotion is approved by the prior generation's authority.
// Misranking, foreign approvals, wrong subjects and refused approvals are
// recorded refusals; degenerate oracle sets are errors.
func TestEvaluatorRanksKnownOutcomesUnderPriorAuthority(t *testing.T) {
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "candidate-evaluator")
	priorAuthority := testutil.ArtifactID(t, artifact.KindEvidence, "generation-0-decider")
	model := func(name string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindModel, name)
	}
	cases := []EvaluatorCase{
		{Model: model("oracle-good-a"), Score: 0.21, KnownGood: true},
		{Model: model("oracle-good-b"), Score: 0.34, KnownGood: true},
		{Model: model("oracle-bad-a"), Score: 4.85, KnownGood: false},
		{Model: model("oracle-bad-b"), Score: 5.69, KnownGood: false},
	}
	commit := "0123456789abcdef0123456789abcdef01234567"
	approval, err := recipe.NewDecision(
		evaluator, recipe.DecisionAccepted, recipe.EvidenceProduction,
		"generation 0 approves the candidate evaluator",
		recipe.Decider{CodeCommit: commit, Derivation: priorAuthority}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	promotion, err := PromoteEvaluator(evaluator, cases, approval, priorAuthority)
	if err != nil {
		t.Fatal(err)
	}
	if !promotion.Promoted || !strings.Contains(promotion.Reason, "margin") {
		t.Fatalf("correct ranking under prior authority refused: %+v", promotion)
	}
	replay, err := PromoteEvaluator(evaluator, cases, approval, priorAuthority)
	if err != nil || replay.ID != promotion.ID {
		t.Fatalf("promotion identity not deterministic: (%v, %v)", replay.ID, err)
	}

	misranked := append([]EvaluatorCase(nil), cases...)
	misranked[0].Score = 6.0 // a known-good model scored worse than the bads
	refused, err := PromoteEvaluator(evaluator, misranked, approval, priorAuthority)
	if err != nil || refused.Promoted || !strings.Contains(refused.Reason, "misranks") {
		t.Fatalf("misranking promoted: (%+v, %v)", refused, err)
	}

	imposterAuthority := testutil.ArtifactID(t, artifact.KindEvidence, "imposter-authority")
	foreign, err := PromoteEvaluator(evaluator, cases, approval, imposterAuthority)
	if err != nil || foreign.Promoted {
		t.Fatalf("approval from a non-prior authority promoted: (%+v, %v)", foreign, err)
	}
	otherSubject, err := recipe.NewDecision(
		testutil.ArtifactID(t, artifact.KindEvidence, "different-evaluator"),
		recipe.DecisionAccepted, recipe.EvidenceProduction, "approves something else",
		recipe.Decider{CodeCommit: commit, Derivation: priorAuthority}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	wrongSubject, err := PromoteEvaluator(evaluator, cases, otherSubject, priorAuthority)
	if err != nil || wrongSubject.Promoted {
		t.Fatalf("approval for another subject promoted: (%+v, %v)", wrongSubject, err)
	}
	rejected, err := recipe.NewDecision(
		evaluator, recipe.DecisionRefused, recipe.EvidenceProduction, "prior authority refuses",
		recipe.Decider{CodeCommit: commit, Derivation: priorAuthority},
		[]artifact.ID{priorAuthority},
	)
	if err != nil {
		t.Fatal(err)
	}
	notAccepted, err := PromoteEvaluator(evaluator, cases, rejected, priorAuthority)
	if err != nil || notAccepted.Promoted {
		t.Fatalf("refused approval promoted: (%+v, %v)", notAccepted, err)
	}

	if _, err := PromoteEvaluator(evaluator, cases[:2], approval, priorAuthority); err == nil {
		t.Fatal("oracle set without known-bad models accepted")
	}

	content, err := promotion.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseEvaluatorPromotion(content.Data)
	if err != nil || parsed.ID != promotion.ID || !parsed.Promoted {
		t.Fatalf("roundtrip = (%+v, %v)", parsed, err)
	}
	batch, err := promotion.Batch("evaluator/" + promotion.ID.String())
	if err != nil || len(batch.Lineage) != 2+len(cases) {
		t.Fatalf("batch lineage = (%d, %v), want evaluator+approval+oracles", len(batch.Lineage), err)
	}
}
