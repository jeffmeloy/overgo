//go:build windows

package inference

import (
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tokenizer"
)

func TestPromptCacheReuseAfterDecodeTokens(t *testing.T) {
	assertPromptCacheAfterDecode(t, openHermeticScoringRunner(t))
}

func TestPromptCacheRetainsDecodedTokensDevice(t *testing.T) {
	assertDecodedTokensContinue(t, openHermeticScoringRunner(t))
}

// TestRecurrentPromptCacheReadsOnFromItsTurn: a hybrid's prompt entry keeps the cache before its last turn, its decode
// appends beside the entry in place, and a follow-up that keeps the turns before it reads on from there to a cold
// read's tokens.
func TestRecurrentPromptCacheReadsOnFromItsTurn(t *testing.T) {
	cudatest.Require(t)
	runner, err := openF32FixtureRunner(writeHermeticQwen35GGUFLast(t, chatMLStart), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	// Token 7 opens a turn: the prompt's history is 1 4 5 and the follow-up keeps it and that turn's opening token.
	prompt, followUp := []tokenizer.TokenID{1, 4, 5, 7, 6, 4}, []tokenizer.TokenID{1, 4, 5, 7, 5, 6, 7, 4}
	// A history shorter than promptBoundaryTokens is read whole: its turn read apart costs more than reading it again.
	generatePromptCacheFixture(t, runner, prompt, true, false)
	for _, boundary := range runner.promptCaches[0].Boundaries {
		if boundary.Tokens < uint32(len(prompt)) {
			t.Fatalf("a three-token history kept a boundary after %d tokens", boundary.Tokens)
		}
	}
	clearPromptCaches(t, runner)
	runner.promptBoundaryTokens = 0
	cold := generatePromptCacheFixture(t, runner, followUp, false, false)
	generatePromptCacheFixture(t, runner, prompt, true, false)
	entry := runner.promptCaches[0]
	if len(entry.Boundaries) == 0 || entry.Boundaries[0].Tokens != 3 {
		t.Fatalf("the decoded entry keeps boundaries %v, want one after 3 tokens", entry.Boundaries)
	}
	for _, boundary := range entry.Boundaries {
		if boundary.storage != entry.Device.storage {
			t.Fatal("the decode copied the cache beside the prompt's entry")
		}
	}
	warm := generatePromptCacheFixture(t, runner, followUp, true, false)
	if warm.evaluation.Cached != 3 || !slices.Equal(cold.ids, warm.ids) {
		t.Fatalf("follow-up read %+v to %v, cold %v", warm.evaluation, warm.ids, cold.ids)
	}
	for token := range cold.logits {
		for index, want := range cold.logits[token] {
			if delta := math.Abs(float64(warm.logits[token][index] - want)); math.IsNaN(delta) || delta > 2e-4 {
				t.Fatalf("token=%d logit=%d delta=%g", token, index, delta)
			}
		}
	}
}
