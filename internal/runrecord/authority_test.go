package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestProposerEvaluatorDeciderAuthorityDomainsAndPriorGeneration pins the
// independent-admission contract: the three roles bind to pairwise-distinct
// authority domains (names AND identities -- distinct artifact IDs alone are
// refused), evaluations name sealed inputs and a clean re-evaluation worker,
// and generation N admits only under generation N-1's decider approval of the
// new evaluator.
func TestProposerEvaluatorDeciderAuthorityDomainsAndPriorGeneration(t *testing.T) {
	identity := func(name string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindEvidence, name)
	}
	proposer := AuthorityDomain{Name: "proposer-workers", Identity: identity("proposer-key")}
	evaluator := AuthorityDomain{Name: "evaluation-plane", Identity: identity("evaluator-key")}
	decider := AuthorityDomain{Name: "promotion-authority", Identity: identity("decider-key")}
	sealed := identity("sealed-inputs")
	clean := identity("clean-worker")

	bootstrap, err := NewAdmissionBinding(AdmissionBinding{
		Generation: 0, Proposer: proposer, Evaluator: evaluator, Decider: decider,
		SealedInputs: sealed, CleanWorker: clean,
	})
	if err != nil {
		t.Fatal(err)
	}
	replay, err := NewAdmissionBinding(AdmissionBinding{
		Generation: 0, Proposer: proposer, Evaluator: evaluator, Decider: decider,
		SealedInputs: sealed, CleanWorker: clean,
	})
	if err != nil || replay.ID != bootstrap.ID {
		t.Fatalf("replay = (%v, %v), want identical identity", replay.ID, err)
	}

	sharedName := evaluator
	sharedName.Name = proposer.Name
	if _, err := NewAdmissionBinding(AdmissionBinding{
		Generation: 0, Proposer: proposer, Evaluator: sharedName, Decider: decider,
		SealedInputs: sealed, CleanWorker: clean,
	}); err == nil {
		t.Fatal("shared domain name accepted")
	}
	sharedIdentity := evaluator
	sharedIdentity.Identity = proposer.Identity
	if _, err := NewAdmissionBinding(AdmissionBinding{
		Generation: 0, Proposer: proposer, Evaluator: sharedIdentity, Decider: decider,
		SealedInputs: sealed, CleanWorker: clean,
	}); err == nil {
		t.Fatal("distinct names with one shared identity accepted; domains are not merely distinct IDs")
	}
	if _, err := NewAdmissionBinding(AdmissionBinding{
		Generation: 0, Proposer: proposer, Evaluator: evaluator, Decider: decider,
		CleanWorker: clean,
	}); err == nil {
		t.Fatal("binding without sealed inputs accepted")
	}
	if _, err := NewAdmissionBinding(AdmissionBinding{
		Generation: 1, Proposer: proposer, Evaluator: evaluator, Decider: decider,
		SealedInputs: sealed, CleanWorker: clean,
	}); err == nil {
		t.Fatal("generation 1 without prior approval accepted")
	}

	nextEvaluator := AuthorityDomain{Name: "evaluation-plane-v2", Identity: identity("evaluator-key-v2")}
	approval, err := recipe.NewDecision(
		nextEvaluator.Identity, recipe.DecisionAccepted, recipe.EvidenceProduction,
		"generation 0 decider approves the generation 1 evaluator",
		recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: decider.Identity},
		[]artifact.ID{sealed},
	)
	if err != nil {
		t.Fatal(err)
	}
	next, err := NewAdmissionBinding(AdmissionBinding{
		Generation: 1, Proposer: proposer, Evaluator: nextEvaluator, Decider: decider,
		SealedInputs: sealed, CleanWorker: clean, PriorApproval: approval.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateAdmissionSuccession(next, bootstrap, approval); err != nil {
		t.Fatalf("legal succession refused: %v", err)
	}

	imposter, err := recipe.NewDecision(
		nextEvaluator.Identity, recipe.DecisionAccepted, recipe.EvidenceProduction,
		"self-approval by the proposer domain",
		recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: proposer.Identity},
		[]artifact.ID{sealed},
	)
	if err != nil {
		t.Fatal(err)
	}
	forged := next
	forged.PriorApproval = imposter.ID
	forged, err = NewAdmissionBinding(AdmissionBinding{
		Generation: 1, Proposer: proposer, Evaluator: nextEvaluator, Decider: decider,
		SealedInputs: sealed, CleanWorker: clean, PriorApproval: imposter.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateAdmissionSuccession(forged, bootstrap, imposter); err == nil {
		t.Fatal("approval decided outside the prior decider authority accepted")
	}
	if err := ValidateAdmissionSuccession(next, bootstrap, imposter); err == nil {
		t.Fatal("binding citing a different approval accepted")
	}
	skipped, err := NewAdmissionBinding(AdmissionBinding{
		Generation: 2, Proposer: proposer, Evaluator: nextEvaluator, Decider: decider,
		SealedInputs: sealed, CleanWorker: clean, PriorApproval: approval.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateAdmissionSuccession(skipped, bootstrap, approval); err == nil {
		t.Fatal("generation skip accepted")
	}

	content, err := next.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseAdmissionBinding(content.Data)
	if err != nil || parsed.ID != next.ID || parsed.PriorApproval != approval.ID {
		t.Fatalf("roundtrip = (%+v, %v)", parsed, err)
	}
}
