//go:build windows && (models || modeltest)

package inference

import (
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestFlashNextHomesEveryExpert opens Flash-Next's round and finds every expert on the card with its copy in host
// memory, then reads a prompt of 262,000 tokens: the open reserves what it allocates with an ordinary commit first,
// which Windows grows its page file for at once, where the driver's own requests fail at the commit limit and leave
// the card's experts without copies that no promotion or adaptive swap may then displace (flash-next-load).
func TestFlashNextHomesEveryExpert(t *testing.T) {
	cudatest.Require(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	model, draft := roots.ResolveModelPath(strataRoundModel), roots.ResolveModelPath(strataDraftModel)
	requireFlashNext(t, model, draft)
	runner := testcheck.Must(openFixtureRunnerWithResidency(model, OpenOptions{Draft: draft, StrataRound: true}, recipe.ResidencyHybridNative))(t)
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	memory := testcheck.Must(runner.DeviceMemory(context.Background()))(t)
	if memory.Experts == nil || memory.Experts.CardExperts == 0 || memory.Experts.CardOnlyExperts != 0 {
		t.Fatalf("the open placed the experts as %+v, want every card expert with its host copy", memory.Experts)
	}
	length := flashNextPromptLengths[len(flashNextPromptLengths)-1]
	prompt, tokens := flashNextPrompt(t, runner, length)
	ids, _, err := runner.Generate(context.Background(), prompt, GenerateOptions{MaxNewTokens: 16})
	if err != nil || len(ids) <= tokens {
		t.Fatalf("a prompt of %d tokens answered %d tokens: %v", tokens, len(ids)-tokens, err)
	}
}
