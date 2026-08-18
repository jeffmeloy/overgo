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

// Residual-injection RETRIAL protocol, written before the run (Probe
// Discipline): the first probe refused on 13 training windows with fit
// residual 17.4 -- a data-starved ridge, mechanism unrefuted. This retrial
// changes exactly one variable: training data volume. A 7,000-token stream
// of real corpus text tokenized by the scorer's own vocabulary supplies an
// order of magnitude more sliding windows at stride 48; the connector,
// feature-matching objective, alpha, ridge, window geometry and displaced
// -window scoring are unchanged from the refused protocol. SHIP iff injected
// CE beats the un-injected baseline on BOTH trailing held-out windows;
// anything else is a measured refusal that this time cannot blame sample
// count. Deterministic: FP64 ridge with partial pivoting, host reference.
const injectionRetrialStride = 48

// TestResidualInjectionRetrial passes when the experiment ran to a legal
// verdict -- SHIP or measured REFUSAL -- and fails only when it could not
// execute. The docs/plan.json row consumes the logged verdict.
func TestResidualInjectionRetrial(t *testing.T) {
	cudatest.RequireProbe(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	scorerDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	drafterDir := filepath.Join(roots.Models, "MiniCPM5-1B")
	for _, directory := range []string{scorerDir, drafterDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; injection retrial NOT verified", directory)
		}
	}
	var tokens []int
	if err := jsonfile.Decode(testutil.FixturePath(t, "injection_retrial_tokens.json"), &tokens); err != nil {
		t.Fatal(err)
	}
	span := injectionPrefix + injectionGap + injectionTarget
	if len(tokens) < 10*span {
		t.Fatalf("retrial stream too short for a data-volume retrial: %d tokens", len(tokens))
	}
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
		t.Fatalf("retrial assembled %d training windows; the point is volume", len(train))
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
	t.Logf("injection retrial verdict: %s", verdict)
	t.Logf("scorer layer %d, drafter layer %d, %d train windows (was 13), fit residual %.6f (was 17.4)",
		result.ScorerLayer, result.DrafterLayer, result.TrainWindows, result.FitResidual)
	for _, outcome := range result.Outcomes {
		t.Logf("held-out %d: baseline CE %.6f injected CE %.6f delta %+.6f",
			outcome.Window, outcome.BaselineCE, outcome.InjectedCE, outcome.BaselineCE-outcome.InjectedCE)
	}
	t.Logf("reason: %s", result.Reason)
	t.Logf("budget: %d train + %d held-out windows, alpha %g ridge %g, wall %s, host reference, deterministic",
		len(train), len(heldOut), injectionAlpha, injectionRidge, time.Since(started))
}
