package plan

import (
	"encoding/json"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestExperimentLeaseLifecycleAndMeasurement pins the runtime-resource
// -scheduler contract: experiment leases carry resource and wall estimates,
// heartbeat expiry, checkpoint and retry identity; reconciliation derives
// abandonment from expiry or measured outcomes, recommends the incremented
// retry with the resume checkpoint and the measured recovery cost, reports
// prediction error for measured leases -- and remains recommendation-only.
func TestExperimentLeaseLifecycleAndMeasurement(t *testing.T) {
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	experiment := testutil.ArtifactID(t, artifact.KindEvidence, "scheduled-experiment")
	checkpoint := testutil.ArtifactID(t, artifact.KindCheckpoint, "scheduled-checkpoint")
	lease := func(name string, expires time.Time, retry uint32) WorkLease {
		return WorkLease{
			Version: workLeaseVersion, Task: name, Worktree: "worktrees/" + name,
			Branch: "lane/" + name, Role: "experiment", TargetHead: "0123456789abcdef0123456789abcdef01234567",
			ConflictsWith: []string{}, Resources: Resources{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 8},
			ExpiresAt: expires.Format(time.RFC3339Nano), Experiment: experiment,
			Checkpoint: checkpoint, Retry: retry, PredictedWallNS: uint64(30 * time.Minute),
			ID: testutil.ArtifactID(t, artifact.KindEvidence, "lease-"+name),
		}
	}
	live := lease("alive", now.Add(time.Hour), 0)
	expired := lease("expired", now.Add(-time.Hour), 1)
	measuredLease := lease("measured", now.Add(-time.Minute), 0)
	abandonedLease := lease("abandoned", now.Add(-time.Minute), 2)
	outcomes := []LeaseOutcome{
		{
			Lease: measuredLease.ID, Predicted: measuredLease.Resources,
			Actual:          Resources{CPUThreads: 6, HostRAMGiB: 12, VRAMGiB: 7},
			PredictedWallNS: measuredLease.PredictedWallNS, ActualWallNS: uint64(45 * time.Minute),
		},
		{
			Lease: abandonedLease.ID, Predicted: abandonedLease.Resources,
			Actual: abandonedLease.Resources, Abandoned: true, RecoveryNS: uint64(5 * time.Minute),
		},
	}
	recommendations := ReconcileExperimentLeases(
		[]WorkLease{live, expired, measuredLease, abandonedLease}, outcomes, now)
	if len(recommendations) != 4 {
		t.Fatalf("recommendations = %d, want one per lease", len(recommendations))
	}
	byTask := map[string]LeaseRecommendation{}
	for _, recommendation := range recommendations {
		byTask[recommendation.Task] = recommendation
	}
	if r := byTask["alive"]; r.Abandoned || r.RecommendedRetry != 0 {
		t.Fatalf("live lease = %+v, want no action", r)
	}
	if r := byTask["expired"]; !r.Abandoned || r.RecommendedRetry != 2 ||
		r.ResumeCheckpoint != checkpoint || r.Experiment != experiment {
		t.Fatalf("expired lease = %+v, want abandonment with retry 2 from the checkpoint", r)
	}
	if r := byTask["abandoned"]; !r.Abandoned || r.RecommendedRetry != 3 ||
		r.RecoveryNS != uint64(5*time.Minute) {
		t.Fatalf("measured abandonment = %+v, want retry 3 with measured recovery cost", r)
	}
	if r := byTask["measured"]; r.Abandoned || r.WallErrorNS != int64(15*time.Minute) {
		t.Fatalf("measured lease = %+v, want +15m wall error and no abandonment", r)
	}

	// Recommendation-only: reconciliation mutates nothing it was given.
	if expired.Retry != 1 || abandonedLease.Retry != 2 {
		t.Fatal("reconciliation mutated lease retry identities")
	}

	// The experiment-lease extension survives the document codec: identify,
	// roundtrip, and reject a checkpoint of the wrong kind.
	specification := live
	specification.ID = artifact.ID{}
	identified, _, err := workLeaseCodec.Normalize(mustJSON(t, specification))
	if err != nil {
		t.Fatal(err)
	}
	if identified.Experiment != experiment || identified.Checkpoint != checkpoint ||
		identified.PredictedWallNS != uint64(30*time.Minute) {
		t.Fatalf("normalized lease = %+v, want experiment extension preserved", identified)
	}
}

func mustJSON(t *testing.T, value WorkLease) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
