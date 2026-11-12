//go:build windows && (models || modeltest)

package inference

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"sort"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// flashNextPromptLengths are the prompts TestFlashNextAnswersEveryPromptLength reads, in tokens: Strata's short fresh
// prompt, a long one, and one 144 tokens short of the model's context of 262,144, which leaves room for its answer.
var flashNextPromptLengths = []int{4096, 65536, 262000}

// TestFlashNextAnswersEveryPromptLength opens Flash-Next's round three times in a row and reads a prompt of every
// length on each open, answering each: the open leaves the machine's commit for the longest prompt before it gives the
// card's experts their host copies (HomeExperts), so no read meets an allocation the machine cannot give, where every
// expert's copy pinned beside the card's share had outrun the commit of the 98 GB machine at the open
// (flash-next-load).
func TestFlashNextAnswersEveryPromptLength(t *testing.T) {
	cudatest.Require(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	model, draft := roots.ResolveModelPath(strataRoundModel), roots.ResolveModelPath(strataDraftModel)
	for _, path := range []string{model, draft} {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("the prompt lengths need %s: %v", path, err)
		}
	}
	for open := range 3 {
		runner, err := openFixtureRunnerWithResidency(model, OpenOptions{Draft: draft, StrataRound: true}, recipe.ResidencyHybridNative)
		if err != nil {
			t.Fatalf("open %d: %v", open+1, err)
		}
		for _, length := range flashNextPromptLengths {
			prompt, tokens := flashNextPrompt(t, runner, length)
			// Generate answers the prompt's ids and those it decoded after them.
			ids, _, err := runner.Generate(context.Background(), prompt, GenerateOptions{MaxNewTokens: 16})
			if err != nil || len(ids) <= tokens {
				t.Errorf("open %d: a prompt of %d tokens (%d asked) answered %d tokens: %v", open+1, tokens, length, len(ids)-tokens, err)
			}
		}
		memory, err := runner.DeviceMemory(context.Background())
		if err == nil && memory.Experts != nil {
			t.Logf("open %d: %d experts on the card, %d with a copy in host memory", open+1, memory.Experts.CardExperts,
				memory.Experts.HostExperts)
		}
		if err := runner.Close(); err != nil {
			t.Fatalf("open %d's close: %v", open+1, err)
		}
	}
}

// flashNextPrompt is a list of words the runner reads as at most length tokens, as many words as fit, and its tokens.
func flashNextPrompt(t *testing.T, runner *Runner, length int) (string, int) {
	t.Helper()
	vocabulary := strings.Fields("river stone mill harbor lantern orchard copper meadow signal ledger anchor willow " +
		"granite thistle beacon furnace compass marble falcon cedar")
	draws := rand.New(rand.NewPCG(uint64(length), 7))
	words := make([]string, length)
	for index := range words {
		words[index] = vocabulary[draws.IntN(len(vocabulary))]
	}
	text := func(count int) string {
		return fmt.Sprintf("Here is a list of %d words:\n%s\n\nName the three words that appear most often.", count, strings.Join(words[:count], " "))
	}
	var failure error
	count := sort.Search(length, func(count int) bool {
		ids, err := runner.TokenizeSamplingText(text(count + 1))
		if err != nil {
			failure = err
			return true
		}
		return len(ids) > length
	})
	testcheck.NoError(t, failure)
	ids := testcheck.Must(runner.TokenizeSamplingText(text(count)))(t)
	return text(count), len(ids)
}
