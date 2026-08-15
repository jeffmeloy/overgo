package plan

import (
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

// AutonomyEvidence is a bounded admission packet for one scheduling class.
// IDs name independently stored evidence; this packet does not promote or
// dispatch anything.
type AutonomyEvidence struct {
	Class              string
	Evaluation         SchedulingEvaluation
	Developer          artifact.ID
	IndependentSQA     artifact.ID
	SealedPromotion    artifact.ID
	Containment        artifact.ID
	Rollback           artifact.ID
	Recovery           artifact.ID
	MeasuredRecoveryNS uint64
}

type AutonomyAdmission struct {
	Class    string
	Admitted bool
	Missing  []string
}

// AssessAutonomy admits only a class whose recommendation won every observed
// comparison and whose independent protection/recovery evidence is complete.
// The result is eligibility advice; owner promotion remains external.
func AssessAutonomy(evidence AutonomyEvidence) AutonomyAdmission {
	decision := AutonomyAdmission{Class: evidence.Class, Missing: []string{}}
	if !textcheck.Bounded(evidence.Class, 2048, "\x00\r\n") {
		decision.Missing = append(decision.Missing, "class")
	}
	if evidence.Evaluation.Samples == 0 || evidence.Evaluation.RecommendationWins != evidence.Evaluation.Samples ||
		evidence.Evaluation.Collisions != 0 || evidence.Evaluation.Abandonments != 0 {
		decision.Missing = append(decision.Missing, "sustained-benefit")
	}
	ids := []artifact.ID{evidence.Developer, evidence.IndependentSQA, evidence.SealedPromotion,
		evidence.Containment, evidence.Rollback, evidence.Recovery}
	if evidence.Developer.Kind() != artifact.KindEvidence || evidence.IndependentSQA.Kind() != artifact.KindEvidence ||
		evidence.Developer == evidence.IndependentSQA {
		decision.Missing = append(decision.Missing, "independent-sqa")
	}
	if !allDistinctEvidence(ids) {
		decision.Missing = append(decision.Missing, "distinct-evidence")
	}
	for _, requirement := range []struct {
		name string
		id   artifact.ID
	}{
		{"sealed-promotion", evidence.SealedPromotion},
		{"containment", evidence.Containment},
		{"rollback", evidence.Rollback},
		{"recovery", evidence.Recovery},
	} {
		if requirement.id.Kind() != artifact.KindEvidence {
			decision.Missing = append(decision.Missing, requirement.name)
		}
	}
	if evidence.MeasuredRecoveryNS == 0 {
		decision.Missing = append(decision.Missing, "measured-recovery")
	}
	decision.Missing = slices.Compact(decision.Missing)
	decision.Admitted = len(decision.Missing) == 0
	return decision
}

func allDistinctEvidence(ids []artifact.ID) bool {
	seen := make(map[artifact.ID]bool, len(ids))
	for _, id := range ids {
		if id.Kind() != artifact.KindEvidence || seen[id] {
			return false
		}
		seen[id] = true
	}
	return true
}
