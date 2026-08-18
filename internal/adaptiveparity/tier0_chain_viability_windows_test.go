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

// Tier-0 chain protocol, written before the run (Probe Discipline): scorer =
// Qwen2.5-0.5B, drafter = MiniCPM5-1B, chained through typed workflowrecipe
// ports -- the drafter's greedy continuation text feeds the scorer's tokenizer
// over a text edge, no latent bridging. Data: the recorded adaptive profile
// token stream sliced into three disjoint windows of prefix 64 + gap 8 +
// held-out 32 scorer tokens. Both arms fill the gap generatively (baseline:
// the scorer drafts its own gap; chain: the drafter's text crosses the
// tokenizer boundary), then the scorer's CE is measured over the identical
// held-out targets. SHIP iff the chain CE lands strictly below the baseline
// CE on every window (envelope separation at n=3); anything else is a REFUSAL
// recorded with the measured CEs. Budget: 3 windows x 2 arms x 8 greedy
// tokens, host reference, deterministic (no seeds), one machine, under the
// verify timeout. The generation record commits either way.
const (
	tier0ChainPrefix = 64
	tier0ChainDraft  = 8
	tier0ChainTarget = 32
)

// TestTier0ChainViability passes when the experiment ran to a legal verdict --
// SHIP or measured REFUSAL -- and fails only when the experiment could not
// execute. The docs/plan.json Phase B row consumes the logged verdict.
func TestTier0ChainViability(t *testing.T) {
	cudatest.RequireProbe(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	scorerDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	drafterDir := filepath.Join(roots.Models, "MiniCPM5-1B")
	for _, directory := range []string{scorerDir, drafterDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; Tier-0 chain viability NOT verified", directory)
		}
	}
	span := tier0ChainPrefix + tier0ChainDraft + tier0ChainTarget
	tokens := qwenAdaptiveProfileTokens
	if len(tokens) < 3*span {
		t.Fatalf("recorded profile stream too short: %d tokens, need %d", len(tokens), 3*span)
	}
	windows := [][]int{
		tokens[:span],
		tokens[span : 2*span],
		tokens[2*span : 3*span],
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	config := composition.ChainConfig{
		ScorerDir: scorerDir, DrafterDir: drafterDir,
		Prefix: tier0ChainPrefix, Draft: tier0ChainDraft, Target: tier0ChainTarget,
		Windows: windows,
	}
	started := time.Now()
	result, err := composition.RunChainViability(store, config)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Outcomes) != len(windows) {
		t.Fatalf("outcomes = %d, want one per window", len(result.Outcomes))
	}
	record, err := composition.RecordChainViability(store, config, result)
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
	t.Logf("tier-0 chain viability verdict: %s", verdict)
	t.Logf("chain recipe %s baseline recipe %s generation record %s",
		result.ChainRecipe, result.BaselineRecipe, record.ID)
	for _, outcome := range result.Outcomes {
		t.Logf("window %d: baseline CE %.6f chain CE %.6f (chain context %d bytes)",
			outcome.Window, outcome.BaselineCE, outcome.ChainCE, len(outcome.ChainText))
	}
	t.Logf("reason: %s", result.Reason)
	t.Logf("budget: %d windows x 2 arms x %d greedy tokens, wall %s, host reference, deterministic",
		len(windows), tier0ChainDraft, time.Since(started))
}
