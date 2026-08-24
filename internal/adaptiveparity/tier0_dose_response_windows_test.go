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
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// Tier-0 dose-response protocol, written before the run (Probe Discipline):
// the shipped chain gain (draft 8, MiniCPM5-1B drafting for Qwen2.5-0.5B) is
// one point; this probe traces the curve. Draft length sweeps {4, 8, 16, 32}
// in BOTH directions -- strong-to-weak (MiniCPM5-1B drafts, Qwen2.5-0.5B
// scores) and weak-to-strong (reversed) -- under the identical window
// protocol as the SHIP (prefix 64, target 32, 3 disjoint windows of the
// recorded stream). Every configuration runs to a legal verdict and commits
// its generation record; the curve and the direction asymmetry ARE the
// result. No envelope gate on individual configs: the law, not any single
// point, is the deliverable. Budget: 8 configs x 3 windows x 2 arms, greedy
// deterministic, host reference, under the verify timeout.
var doseResponseDrafts = []int{4, 8, 16, 32}

const (
	dosePrefix = 64
	doseTarget = 32
)

// TestTier0ChainDoseResponse passes when every configuration ran to a legal
// verdict with its generation record committed; it fails only when a
// configuration could not execute.
func TestTier0ChainDoseResponse(t *testing.T) {
	cudatest.RequireProbe(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	qwen := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	minicpm := filepath.Join(roots.Models, "MiniCPM5-1B")
	for _, directory := range []string{qwen, minicpm} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; dose-response NOT verified", directory)
		}
	}
	tokens := qwenAdaptiveProfileTokens
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	directions := []struct {
		name            string
		scorer, drafter string
	}{
		{"strong-to-weak", qwen, minicpm},
		{"weak-to-strong", minicpm, qwen},
	}
	started := time.Now()
	for _, direction := range directions {
		for _, draft := range doseResponseDrafts {
			span := dosePrefix + draft + doseTarget
			if len(tokens) < 3*span {
				t.Fatalf("stream too short for draft %d: %d < %d", draft, len(tokens), 3*span)
			}
			config := composition.ChainConfig{
				ScorerDir: direction.scorer, DrafterDir: direction.drafter,
				Prefix: dosePrefix, Draft: draft, Target: doseTarget,
				Windows: [][]int{
					tokens[:span], tokens[span : 2*span], tokens[2*span : 3*span],
				},
			}
			result, err := composition.RunChainViability(store, config)
			if err != nil {
				t.Fatalf("%s draft %d: %v", direction.name, draft, err)
			}
			if len(result.Outcomes) != 3 {
				t.Fatalf("%s draft %d: %d outcomes", direction.name, draft, len(result.Outcomes))
			}
			record, err := composition.RecordChainViability(store, config, result)
			if err != nil {
				t.Fatalf("%s draft %d record: %v", direction.name, draft, err)
			}
			meanDelta := 0.0
			for _, outcome := range result.Outcomes {
				meanDelta += (outcome.BaselineCE - outcome.ChainCE) / 3
			}
			verdict := "REFUSE"
			if result.Ship {
				verdict = "SHIP"
			}
			t.Logf("%s draft=%2d verdict=%s mean-CE-gain=%+.6f record=%s",
				direction.name, draft, verdict, meanDelta, record.ID)
		}
	}
	t.Logf("dose-response budget: %d configs x 3 windows x 2 arms, wall %s, host reference, deterministic",
		len(directions)*len(doseResponseDrafts), time.Since(started))
}
