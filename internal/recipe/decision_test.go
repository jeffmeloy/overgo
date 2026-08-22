package recipe

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

const decisionTestCommit = "0123456789abcdef0123456789abcdef01234567"

func TestDecisionRoundTripIncludesDeciderAndTier(t *testing.T) {
	subject := testutil.ArtifactID(t, artifact.KindRecipe, "subject")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "derivation")
	evidence := testutil.ArtifactID(t, artifact.KindRun, "run")
	decision, err := NewDecision(
		subject, DecisionRefused, EvidenceParity, "output parity failed",
		Decider{CodeCommit: decisionTestCommit, Derivation: derivation},
		[]artifact.ID{derivation, evidence},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseDecision(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != decision.ID || parsed.Decider != decision.Decider || parsed.Tier != EvidenceParity ||
		parsed.Reason != decision.Reason {
		t.Fatalf("parsed decision = %+v", parsed)
	}
	batch, err := decision.Batch("test/decision")
	if err != nil {
		t.Fatal(err)
	}
	parents := map[artifact.ID]bool{subject: true, derivation: true, evidence: true}
	if len(batch.Contents) != 1 || len(batch.Lineage) != len(parents) {
		t.Fatalf("decision batch = %+v", batch)
	}
	for _, edge := range batch.Lineage {
		if edge.Child != decision.ID || edge.Relation != artifact.RelationDependsOn || !parents[edge.Parent] {
			t.Fatalf("decision lineage = %+v", batch.Lineage)
		}
	}
	second, err := NewDecision(
		subject, DecisionRefused, EvidenceExperimental, decision.Reason,
		decision.Decider, decision.Evidence,
	)
	if err != nil {
		t.Fatal(err)
	}
	if second.ID == decision.ID {
		t.Fatal("decision tier did not affect identity")
	}
	mutated := append([]byte(nil), content.Data...)
	mutated = bytes.Replace(mutated, []byte("parity"), []byte("bogus!"), 1)
	if _, err := ParseDecision(mutated); err == nil {
		t.Fatal("invalid decision tier accepted")
	}
}

// TestRefusalDecisionCarriesMeasuredReason pins the refusal-ledger contract: a
// refused decision must bind both a human-readable reason AND at least one
// measurement evidence identity, so every ledger row traces to what was
// actually measured. Prose-only refusals and evidence-free refusals are
// rejected; non-refusal outcomes keep their existing evidence latitude.
func TestRefusalDecisionCarriesMeasuredReason(t *testing.T) {
	subject := testutil.ArtifactID(t, artifact.KindRecipe, "subject")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "derivation")
	failedRun := testutil.ArtifactID(t, artifact.KindRun, "failed-run")
	decider := Decider{CodeCommit: decisionTestCommit, Derivation: derivation}

	decision, err := NewDecision(
		subject, DecisionRefused, EvidenceExperimental,
		"held-out loss regressed beyond the noise envelope",
		decider, []artifact.ID{failedRun},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseDecision(content.Data)
	if err != nil || parsed.Reason != decision.Reason || len(parsed.Evidence) != 1 || parsed.Evidence[0] != failedRun {
		t.Fatalf("parsed refusal = (%+v, %v)", parsed, err)
	}

	if _, err := NewDecision(
		subject, DecisionRefused, EvidenceExperimental, "prose without measurement",
		decider, nil,
	); err == nil {
		t.Fatal("refusal without measurement evidence accepted")
	}
	if _, err := NewDecision(
		subject, DecisionRefused, EvidenceExperimental, "",
		decider, []artifact.ID{failedRun},
	); err == nil {
		t.Fatal("refusal without reason accepted")
	}
	if _, err := NewDecision(
		subject, DecisionObserved, EvidenceExperimental, "observation needs no measurement binding",
		decider, nil,
	); err != nil {
		t.Fatalf("non-refusal outcome lost its evidence latitude: %v", err)
	}
}

func TestDecisionRejectsMissingReasonAndDecider(t *testing.T) {
	subject := testutil.ArtifactID(t, artifact.KindRecipe, "subject")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "derivation")
	if _, err := NewDecision(
		subject, DecisionRefused, EvidenceExperimental, "",
		Decider{CodeCommit: decisionTestCommit, Derivation: derivation}, nil,
	); err == nil {
		t.Fatal("refusal without reason accepted")
	}
	if _, err := NewDecision(
		subject, DecisionObserved, EvidenceExperimental, "",
		Decider{Derivation: derivation}, nil,
	); err == nil {
		t.Fatal("decision without code commit accepted")
	}
}

func TestDecisionReasonHasNoIndependentLengthPolicy(t *testing.T) {
	subject := testutil.ArtifactID(t, artifact.KindRecipe, "long-reason-subject")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "long-reason-derivation")
	reason := strings.Repeat("measured evidence remains canonical.", len(DecisionMediaType)*len(DecisionSchema))
	if _, err := NewDecision(
		subject, DecisionObserved, EvidenceExperimental, reason,
		Decider{CodeCommit: decisionTestCommit, Derivation: derivation}, nil,
	); err != nil {
		t.Fatalf("canonical evidence reason was rejected by an independent length policy: %v", err)
	}
}
