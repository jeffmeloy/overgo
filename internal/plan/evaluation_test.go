package plan

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestSchedulingRecommendationEvaluation(t *testing.T) {
	first := testutil.ArtifactID(t, artifact.KindEvidence, "lease-one")
	second := testutil.ArtifactID(t, artifact.KindEvidence, "lease-two")
	leases := map[artifact.ID]WorkLease{
		first:  {ID: first, Resources: Resources{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 8}},
		second: {ID: second, Resources: Resources{CPUThreads: 4, HostRAMGiB: 12, VRAMGiB: 4}},
	}
	outcomes := []LeaseOutcome{
		{Lease: first, Predicted: Resources{CPUThreads: 6, HostRAMGiB: 14, VRAMGiB: 7}, Actual: Resources{CPUThreads: 6, HostRAMGiB: 14, VRAMGiB: 7}, PredictedWallNS: 90, ActualWallNS: 100, PredictedInterferenceNS: 5, ActualInterferenceNS: 10},
		{Lease: second, Predicted: leases[second].Resources, Actual: Resources{CPUThreads: 5, HostRAMGiB: 10, VRAMGiB: 4}, PredictedWallNS: 100, ActualWallNS: 130, Collision: true, RecoveryNS: 20},
	}
	evaluation, err := EvaluateScheduling(outcomes, leases)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.Samples != 2 || evaluation.RecommendationAgreements != 1 || evaluation.RecommendationWins != 1 ||
		evaluation.Ties != 1 || evaluation.Collisions != 1 || evaluation.TotalRecoveryNS != 20 ||
		evaluation.MaxWallErrorNS != 30 || evaluation.MaxInterferenceErrorNS != 5 {
		t.Fatalf("evaluation = %+v", evaluation)
	}
	if _, err := EvaluateScheduling(outcomes, map[artifact.ID]WorkLease{}); err == nil {
		t.Fatal("evaluation accepted missing owner decisions")
	}
}
