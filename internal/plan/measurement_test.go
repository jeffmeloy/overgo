package plan

import (
	"encoding/json"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/worklease"
)

func TestLeaseOutcomeMetrics(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	leaseData, _ := json.Marshal(worklease.Lease{
		Version: worklease.Version, Task: "task", Worktree: "C:/worktree", Branch: "codex/task", Role: "developer",
		TargetHead: "0123456789abcdef0123456789abcdef01234567", ConflictsWith: []string{},
		Resources: worklease.Resources{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 8}, ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano),
	})
	lease, err := worklease.Record(t.Context(), store, leaseData)
	if err != nil {
		t.Fatal(err)
	}
	outcomeData, _ := json.Marshal(LeaseOutcome{
		Version: leaseOutcomeVersion, Lease: lease.ID, Predicted: lease.Resources,
		Actual:          worklease.Resources{CPUThreads: 7, HostRAMGiB: 14, VRAMGiB: 7},
		PredictedWallNS: 100, ActualWallNS: 120, PredictedInterferenceNS: 10, ActualInterferenceNS: 20,
		Collision: true, RecoveryNS: 5,
	})
	outcome, err := RecordLeaseOutcome(t.Context(), store, outcomeData)
	if err != nil {
		t.Fatal(err)
	}
	parsed, ok, err := ReadLeaseOutcome(t.Context(), store, outcome.ID)
	if err != nil || !ok || parsed.ActualWallNS != 120 || parsed.RecoveryNS != 5 {
		t.Fatalf("lease outcome = (%+v, %v, %v)", parsed, ok, err)
	}
}
