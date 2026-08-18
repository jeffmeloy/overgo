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
	"overgo/internal/jsonfile"
	"overgo/internal/testutil"
)

// Linear-attention connector protocol, written before the run (Probe
// Discipline): the mean-vector ridge connector is refuted at adequate data
// volume -- injection lost on both held-out windows with 140 training
// samples. This variant changes exactly one mechanism: content addressing.
// Each target position retrieves its own injection vector by linear
// attention (elu+1 kernel) over the drafter's per-token states projected
// through the SAME ridge fit; queries are the scorer's own un-injected
// hidden states. No new fitted parameters -- if this ships, content
// addressing was what the linear connector lacked; if it refuses, the
// linear connector family is refuted at this budget with or without
// attention, and typed interfaces keep the field.
func TestLinearAttentionConnectorViability(t *testing.T) {
	cudatest.RequireProbe(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	scorerDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	drafterDir := filepath.Join(roots.Models, "MiniCPM5-1B")
	for _, directory := range []string{scorerDir, drafterDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; linear-attention connector NOT verified", directory)
		}
	}
	var tokens []int
	if err := jsonfile.Decode(testutil.FixturePath(t, "injection_retrial_tokens.json"), &tokens); err != nil {
		t.Fatal(err)
	}
	span := injectionPrefix + injectionGap + injectionTarget
	heldOutStart := len(tokens) - 2*span
	var train [][]int
	for start := 0; start+span <= heldOutStart; start += injectionRetrialStride {
		train = append(train, tokens[start:start+span])
	}
	heldOut := [][]int{
		tokens[heldOutStart : heldOutStart+span],
		tokens[heldOutStart+span : heldOutStart+2*span],
	}
	if len(train) < 100 {
		t.Fatalf("assembled %d training windows; the retrial data volume is the control", len(train))
	}
	started := time.Now()
	result, err := composition.RunLinearAttentionViability(composition.InjectionConfig{
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
	t.Logf("linear-attention connector verdict: %s", verdict)
	t.Logf("scorer layer %d, drafter layer %d, %d train windows, fit residual %.6f",
		result.ScorerLayer, result.DrafterLayer, result.TrainWindows, result.FitResidual)
	for _, outcome := range result.Outcomes {
		t.Logf("held-out %d: baseline CE %.6f injected CE %.6f delta %+.6f (mean-vector retrial was %+0.6f)",
			outcome.Window, outcome.BaselineCE, outcome.InjectedCE, outcome.BaselineCE-outcome.InjectedCE,
			[]float64{-0.159174, -0.597241}[outcome.Window])
	}
	t.Logf("reason: %s", result.Reason)
	t.Logf("budget: %d train + %d held-out windows, alpha %g ridge %g, wall %s, host reference, deterministic",
		len(train), len(heldOut), injectionAlpha, injectionRidge, time.Since(started))
}
