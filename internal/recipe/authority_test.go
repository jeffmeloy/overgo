package recipe

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestArtifactAuthorityReceiptContract(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "candidate model")
	dataset := testutil.ArtifactID(t, artifact.KindDatasetShard, "promotion split")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "compiled recipe")
	evaluation := testutil.ArtifactID(t, artifact.KindEvaluation, "held-out evaluation")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "sealed decision policy")
	run := testutil.ArtifactID(t, artifact.KindRun, "evaluation run")

	first, err := NewAuthoritySubject(model, dataset, recipeID, evaluation)
	if err != nil {
		t.Fatal(err)
	}
	reordered, err := NewAuthoritySubject(evaluation, recipeID, model, dataset)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != reordered.ID || !slices.Equal(first.Members, reordered.Members) {
		t.Fatalf("composite subject is order-dependent: %s != %s", first.ID, reordered.ID)
	}
	content, err := first.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseAuthoritySubject(content.Data)
	if err != nil || parsed.ID != first.ID || len(parsed.Lineage()) != len(first.Members) {
		t.Fatalf("authority subject round trip = (%+v, %v)", parsed, err)
	}

	receipt, err := NewDecision(
		first.ID, DecisionAccepted, EvidenceProduction, "exact promotion subject verified",
		Decider{CodeCommit: decisionTestCommit, Derivation: derivation}, []artifact.ID{run},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := AdmitAuthorityReceipt(first, receipt, EvidenceParity); err != nil {
		t.Fatalf("exact stronger receipt refused: %v", err)
	}

	changed, err := NewAuthoritySubject(model, dataset, recipeID)
	if err != nil {
		t.Fatal(err)
	}
	if err := AdmitAuthorityReceipt(changed, receipt, EvidenceParity); err == nil {
		t.Fatal("receipt admitted a subject with removed evaluation evidence")
	}
	weaker, err := NewDecision(
		first.ID, DecisionAccepted, EvidenceExperimental, "smoke evidence only",
		receipt.Decider, []artifact.ID{run},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := AdmitAuthorityReceipt(first, weaker, EvidenceParity); err == nil {
		t.Fatal("weak receipt admitted at a stronger evidence tier")
	}
	bare, err := NewDecision(
		first.ID, DecisionAccepted, EvidenceProduction, "unmeasured approval",
		receipt.Decider, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := AdmitAuthorityReceipt(first, bare, EvidenceParity); err == nil {
		t.Fatal("evidence-free approval admitted as authority")
	}
	refused, err := NewDecision(
		first.ID, DecisionRefused, EvidenceProduction, "promotion criteria failed",
		receipt.Decider, []artifact.ID{run},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := AdmitAuthorityReceipt(first, refused, EvidenceParity); err == nil {
		t.Fatal("refusal admitted as authorization")
	}
	if _, err := NewAuthoritySubject(model, model); err == nil {
		t.Fatal("ambiguous duplicate subject member accepted")
	}

	mutated := receipt
	mutated.Reason = "mutated after identity"
	if err := AdmitAuthorityReceipt(first, mutated, EvidenceParity); err == nil {
		t.Fatal("mutated receipt identity accepted")
	}
}
