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
	"overgo/internal/testutil"
)

// Composition-viability protocol, written before the run (Probe Discipline):
// donor = Carbon-500M middle-layer MLP (organ-classified mlp_gate), target =
// Qwen2.5-0.5B, graft at the target's middle layer behind a zero-initialized
// Up bridge. Bridge-only Muon, derived learning rate, mu 0.9. Data: the
// recorded adaptive profile token stream sliced into two disjoint train
// windows and one held-out window. SHIP iff every seed's held-out CE lands
// strictly below the unmodified baseline (envelope separation at n=3);
// anything else is a REFUSAL recorded with the measured losses. Budget: 3
// seeds x 6 steps, host reference, one machine, under the verify timeout.
const (
	compositionWindow = 64
	compositionSteps  = 6
)

var compositionSeeds = []int64{7, 11, 13}

// TestComponentCompositionViability is THE PIVOT: it passes when the
// experiment ran to a legal verdict -- SHIP or measured REFUSAL -- and fails
// only when the experiment could not execute. The decision_point in
// docs/plan.json consumes the logged verdict.
func TestComponentCompositionViability(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	targetDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	donorDir := filepath.Join(roots.Models, "Carbon-500M")
	for _, directory := range []string{targetDir, donorDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; composition viability NOT verified", directory)
		}
	}
	tokens := qwenAdaptiveProfileTokens
	if len(tokens) < 3*compositionWindow {
		t.Fatalf("recorded profile stream too short: %d tokens", len(tokens))
	}
	started := time.Now()
	result, err := composition.RunViability(composition.Config{
		TargetDir: targetDir, DonorDir: donorDir,
		GraftLayer: -1, DonorLayer: -1,
		Seeds: compositionSeeds, Steps: compositionSteps,
		BaseLR: 0, Momentum: 0.9,
		Train: [][]int{
			tokens[:compositionWindow],
			tokens[compositionWindow : 2*compositionWindow],
		},
		HeldOut: tokens[2*compositionWindow : 3*compositionWindow],
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Outcomes) != len(compositionSeeds) {
		t.Fatalf("outcomes = %d, want one per seed", len(result.Outcomes))
	}
	for _, outcome := range result.Outcomes {
		if len(outcome.TrainLosses) != compositionSteps {
			t.Fatalf("seed %d ran %d steps, want %d", outcome.Seed, len(outcome.TrainLosses), compositionSteps)
		}
	}
	verdict := "REFUSE"
	if result.Ship {
		verdict = "SHIP"
	}
	t.Logf("composition viability verdict: %s", verdict)
	t.Logf("donor %s (%s) grafted at target layer %d; baseline held-out %.6f",
		result.DonorTensor, result.DonorContract.Role, result.GraftLayer, result.Baseline)
	for _, outcome := range result.Outcomes {
		t.Logf("seed %d: first train %.6f last train %.6f held-out %.6f",
			outcome.Seed, outcome.TrainLosses[0], outcome.TrainLosses[len(outcome.TrainLosses)-1], outcome.HeldOut)
	}
	t.Logf("reason: %s", result.Reason)
	t.Logf("budget: %d seeds x %d steps, wall %s, host reference", len(compositionSeeds), compositionSteps, time.Since(started))
}
