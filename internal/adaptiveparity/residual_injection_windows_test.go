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

// Residual-injection protocol, written before the run (Probe Discipline):
// drafter MiniCPM5-1B's mean hidden state at its 0.75-depth layer, computed
// over its own tokenization of the prefix text, projects through a ridge-fit
// linear connector into scorer Qwen2.5-0.5B's residual stream at the scorer's
// 0.75-depth layer with magnitude-rescaled injection (alpha 0.25, the frozen
// -graph literature's default). The connector trains by FEATURE MATCHING
// against the scorer's measured context-gap vectors -- never end-to-end task
// CE, the recorded failure mode of the retired adapter tier -- on 13 sliding
// windows (stride 16) from the front of the recorded stream. Held-out: the 2
// disjoint trailing windows (prefix 64, gap 8, target 32), displaced-window
// scoring identical to the Tier-0 chain. SHIP iff injected CE beats the
// un-injected baseline on BOTH held-out windows; anything else is a measured
// refusal. The Tier-0 chain's draft-8 gain (+0.171 mean CE) is the reference
// composition mechanism this probe challenges. Deterministic: FP64 ridge with
// partial pivoting, greedy-free evaluation, host reference.
const (
	injectionPrefix = 64
	injectionGap    = 8
	injectionTarget = 32
	injectionStride = 16
	injectionAlpha  = 0.25
	injectionRidge  = 1e-3
)

// TestResidualInjectionConnector passes when the experiment ran to a legal
// verdict -- SHIP or measured REFUSAL -- and fails only when it could not
// execute. The docs/plan.json row consumes the logged verdict.
func TestResidualInjectionConnector(t *testing.T) {
	cudatest.RequireProbe(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	scorerDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	drafterDir := filepath.Join(roots.Models, "MiniCPM5-1B")
	for _, directory := range []string{scorerDir, drafterDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; residual injection NOT verified", directory)
		}
	}
	tokens := qwenAdaptiveProfileTokens
	span := injectionPrefix + injectionGap + injectionTarget
	if len(tokens) < 2*span+span {
		t.Fatalf("stream too short: %d tokens", len(tokens))
	}
	heldOutStart := len(tokens) - 2*span
	var train [][]int
	for start := 0; start+span <= heldOutStart; start += injectionStride {
		train = append(train, tokens[start:start+span])
	}
	heldOut := [][]int{
		tokens[heldOutStart : heldOutStart+span],
		tokens[heldOutStart+span : heldOutStart+2*span],
	}
	started := time.Now()
	result, err := composition.RunInjectionViability(composition.InjectionConfig{
		ScorerDir: scorerDir, DrafterDir: drafterDir,
		Prefix: injectionPrefix, Gap: injectionGap, Target: injectionTarget,
		Alpha: injectionAlpha, Ridge: injectionRidge,
		Train: train, HeldOut: heldOut,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Outcomes) != len(heldOut) {
		t.Fatalf("outcomes = %d, want one per held-out window", len(result.Outcomes))
	}
	verdict := "REFUSE"
	if result.Ship {
		verdict = "SHIP"
	}
	t.Logf("residual injection verdict: %s", verdict)
	t.Logf("scorer layer %d, drafter layer %d, %d train windows, fit residual %.6f",
		result.ScorerLayer, result.DrafterLayer, result.TrainWindows, result.FitResidual)
	for _, outcome := range result.Outcomes {
		t.Logf("held-out %d: baseline CE %.6f injected CE %.6f delta %+.6f",
			outcome.Window, outcome.BaselineCE, outcome.InjectedCE, outcome.BaselineCE-outcome.InjectedCE)
	}
	t.Logf("reason: %s", result.Reason)
	t.Logf("budget: %d train + %d held-out windows, alpha %g ridge %g, wall %s, host reference, deterministic",
		len(train), len(heldOut), injectionAlpha, injectionRidge, time.Since(started))
}
