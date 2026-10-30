//go:build windows && (models || modeltest)

package inference

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestFlashNextRoundIsReadyAtOpen holds a round runner's open to what Strata's start-up makes, so a first request
// after load does not: the round's stores with its converted weights and CPU pool, and the tier, its route counts and
// its PCIe share from the link's probe. The drafter is the first drafting request's (readyStrataRound).
func TestFlashNextRoundIsReadyAtOpen(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	model, draft := roots.ResolveModelPath(strataRoundModel), roots.ResolveModelPath(strataDraftModel)
	runner, err := openFixtureRunnerWithResidency(model, OpenOptions{Draft: draft, StrataRound: true}, recipe.ResidencyHybridNative)
	if err != nil {
		t.Skipf("the round's readiness needs %s and %s: %v", model, draft, err)
	}
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	round := runner.strata
	if round == nil || round.cpu == nil || round.bf16 == nil || round.f16 == nil {
		t.Fatalf("the runner opened its round as %+v, want its stores, converted weights and CPU pool made", round)
	}
	if round.usage == nil || !(round.share > 0) {
		t.Fatalf("the runner opened its round's tier with route counts %v and PCIe share %v, want both made", round.usage != nil, round.share)
	}
}
