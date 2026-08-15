package plan

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestAutonomyAdmissionRequiresRecoveryEvidence(t *testing.T) {
	id := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvidence, name) }
	packet := AutonomyEvidence{
		Class: "cpu-test-lane", Evaluation: SchedulingEvaluation{Samples: 3, RecommendationWins: 3},
		Developer: id("developer"), IndependentSQA: id("sqa"), SealedPromotion: id("promotion"),
		Containment: id("containment"), Rollback: id("rollback"), Recovery: id("recovery"), MeasuredRecoveryNS: 10,
	}
	if decision := AssessAutonomy(packet); !decision.Admitted || len(decision.Missing) != 0 {
		t.Fatalf("complete packet refused: %+v", decision)
	}
	packet.MeasuredRecoveryNS = 0
	if decision := AssessAutonomy(packet); decision.Admitted || !slices.Contains(decision.Missing, "measured-recovery") {
		t.Fatalf("unmeasured recovery admitted: %+v", decision)
	}
	packet.MeasuredRecoveryNS = 10
	packet.IndependentSQA = packet.Developer
	if decision := AssessAutonomy(packet); decision.Admitted || !slices.Contains(decision.Missing, "independent-sqa") {
		t.Fatalf("self-review admitted: %+v", decision)
	}
	packet.IndependentSQA = id("sqa-restored")
	packet.Evaluation.RecommendationWins = 2
	if decision := AssessAutonomy(packet); decision.Admitted || !slices.Contains(decision.Missing, "sustained-benefit") {
		t.Fatalf("mixed benefit admitted: %+v", decision)
	}
}
