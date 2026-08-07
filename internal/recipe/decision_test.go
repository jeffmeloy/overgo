package recipe

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
)

const decisionTestCommit = "0123456789abcdef0123456789abcdef01234567"

func TestDecisionRoundTripIncludesDeciderAndTier(t *testing.T) {
	subject, _ := artifact.IdentifyBytes(artifact.KindRecipe, []byte("subject"))
	derivation, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("derivation"))
	evidence, _ := artifact.IdentifyBytes(artifact.KindRun, []byte("run"))
	decision, err := NewDecision(
		subject, DecisionRefused, EvidenceParity, "output parity failed",
		Decider{CodeCommit: decisionTestCommit, Derivation: derivation},
		[]artifact.ID{evidence},
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

func TestDecisionRejectsMissingReasonAndDecider(t *testing.T) {
	subject, _ := artifact.IdentifyBytes(artifact.KindRecipe, []byte("subject"))
	derivation, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("derivation"))
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
