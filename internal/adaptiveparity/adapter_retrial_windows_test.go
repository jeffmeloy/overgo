//go:build windows

package adaptiveparity_test

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/composition"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// Adapter-retrial protocol, written before the run (Probe Discipline): the
// pivot's measured failure mode was Muon step-size dominance at the 6-step
// budget -- training loss rose under the derived learning rate. This NEW
// recorded protocol reruns the identical graft (donor Carbon-500M middle MLP
// into frozen Qwen2.5-0.5B behind the zero-initialized bridge, same recorded
// token windows) at one tenth the derived step size and three times the step
// budget. SHIP iff every seed's held-out CE lands strictly below the
// unmodified baseline (envelope separation at n=3); anything else is the
// tier's SECOND measured refusal, which retires the adapter tier per the
// plan row. Budget: 3 seeds x 18 steps x lr-scale 0.1, host reference, one
// machine, under the verify timeout. The generation record commits either
// way, distinct from the pivot's by its protocol identity.
const (
	retrialWindow  = 64
	retrialSteps   = 18
	retrialLRScale = 0.1
)

var retrialSeeds = []int64{7, 11, 13}

// TestAdapterRetrialProbe passes when the retrial ran to a legal verdict --
// SHIP or measured REFUSAL -- and fails only when the experiment could not
// execute. The docs/plan.json row consumes the logged verdict.
func TestAdapterRetrialProbe(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	targetDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	donorDir := filepath.Join(roots.Models, "Carbon-500M")
	for _, directory := range []string{targetDir, donorDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; adapter retrial NOT verified", directory)
		}
	}
	tokens := qwenAdaptiveProfileTokens
	if len(tokens) < 3*retrialWindow {
		t.Fatalf("recorded profile stream too short: %d tokens", len(tokens))
	}
	config := composition.Config{
		TargetDir: targetDir, DonorDir: donorDir,
		GraftLayer: -1, DonorLayer: -1,
		Seeds: retrialSeeds, Steps: retrialSteps,
		BaseLR: 0, LRScale: retrialLRScale, Momentum: 0.9,
		Train: [][]int{
			tokens[:retrialWindow],
			tokens[retrialWindow : 2*retrialWindow],
		},
		HeldOut: tokens[2*retrialWindow : 3*retrialWindow],
	}
	started := time.Now()
	result, err := composition.RunViability(config)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Outcomes) != len(retrialSeeds) {
		t.Fatalf("outcomes = %d, want one per seed", len(result.Outcomes))
	}
	for _, outcome := range result.Outcomes {
		if len(outcome.TrainLosses) != retrialSteps {
			t.Fatalf("seed %d ran %d steps, want %d", outcome.Seed, len(outcome.TrainLosses), retrialSteps)
		}
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	record, err := composition.RecordViability(store, config, result)
	if err != nil {
		t.Fatal(err)
	}
	if record.Outcome != runrecord.OutcomeSucceeded && record.Outcome != runrecord.OutcomeFailed {
		t.Fatalf("generation record outcome = %q", record.Outcome)
	}
	verdict := "REFUSE"
	if result.Ship {
		verdict = "SHIP"
	}
	t.Logf("adapter retrial verdict: %s", verdict)
	t.Logf("donor %s (%s) at target layer %d; baseline held-out %.6f; generation record %s",
		result.DonorTensor, result.DonorContract.Role, result.GraftLayer, result.Baseline, record.ID)
	for _, outcome := range result.Outcomes {
		t.Logf("seed %d: first train %.6f last train %.6f held-out %.6f",
			outcome.Seed, outcome.TrainLosses[0], outcome.TrainLosses[len(outcome.TrainLosses)-1], outcome.HeldOut)
	}
	t.Logf("reason: %s", result.Reason)
	t.Logf("budget: %d seeds x %d steps at lr-scale %g, wall %s, host reference",
		len(retrialSeeds), retrialSteps, retrialLRScale, time.Since(started))
}
