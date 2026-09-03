package inference

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/tokenizer"
)

// pagePairContext is the fixture context for the page pair: a 16-token
// past on a 16-token page and a 24-token past on a 32-token page both
// append into the 32-token target page, and sixteen tokens of the
// fixture's key width fill a pool bucket exactly, so a session compiled
// over the 32-token page would read past the 16-token page's buffer.
const pagePairContext = uint32(64)

// TestDecodeSessionServesOnlyItsOwnPastPage drives the page pair that
// faulted the BBH chat pass through the real device generation path on the
// hermetic fixture: a past beyond a capacity band (on the next page) and a
// past exactly on the band (on its own page) both append into the same
// target page. The decode session cached by the larger past must not serve
// the smaller one: the on-band decode selects the token a fresh runner
// selects, and every step completes. The beyond-band prompt decodes first
// because that is the order in which the cached session over-reads.
func TestDecodeSessionServesOnlyItsOwnPastPage(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := writeHermeticLlamaGGUFWithContext(t, pagePairContext)
	const band = 16
	overhang := band / 2
	reference := openPagePairRunner(t, path)
	clean := decodePagePairOnce(t, reference, band)
	if err := reference.Close(); err != nil {
		t.Fatal(err)
	}
	runner := openPagePairRunner(t, path)
	defer runner.Close()
	decodePagePairOnce(t, runner, band+overhang)
	if after := decodePagePairOnce(t, runner, band); after != clean {
		t.Fatalf("on-band decode selected %d after the beyond-band past, %d on a fresh runner", after, clean)
	}
	decodePagePairOnce(t, runner, band+overhang)
	decodePagePairOnce(t, runner, band)
}

func openPagePairRunner(t *testing.T, path string) *Runner {
	t.Helper()
	runner, err := openFixtureRunnerWithResidency(path, OpenOptions{}, recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	if runner.spec.ContextLength != pagePairContext {
		t.Fatalf("fixture context = %d, want %d", runner.spec.ContextLength, pagePairContext)
	}
	return runner
}

// decodePagePairOnce prefills a prompt of the given length and takes one
// greedy decode step from it, the parameterized append whose session the
// runner caches, returning the selected token.
func decodePagePairOnce(t *testing.T, runner *Runner, length int) tokenizer.TokenID {
	t.Helper()
	ctx := t.Context()
	tokens := make([]tokenizer.TokenID, length)
	for index := range tokens {
		tokens[index] = tokenizer.TokenID(1 + index%4)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	past, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedy, tokens, nil)
	if err != nil {
		t.Fatalf("prefill of %d tokens: %v", length, err)
	}
	defer past.Release(ctx)
	if past.Tokens != uint32(length) {
		t.Fatalf("prefill of %d tokens left %d in the cache", length, past.Tokens)
	}
	next, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedy, []tokenizer.TokenID{past.Selected}, past)
	if err != nil {
		t.Fatalf("decode step after %d tokens: %v", length, err)
	}
	defer next.Release(ctx)
	return next.Selected
}
