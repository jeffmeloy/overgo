//go:build windows

package inference

import (
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tokenizer"
)

// TestSpanDecodeSession proves span verify forwards ride the parameterized
// decode-session replay: the first span of a given shape compiles a session,
// later spans of that shape replay it without recompilation, and replayed
// selections match a teacher-forced host oracle.
func TestSpanDecodeSession(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := writeHermeticQwen35GGUF(t)
	runner, err := openF32FixtureRunner(path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()

	// Teacher-forced host oracle over the full token history.
	tokens := []tokenizer.TokenID{1, 4, 5, 6}
	oracle, err := runner.NewContinuousBatch(ContinuousBatchOptions{MaxSequences: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer oracle.Close(context.Background())
	want := make([]tokenizer.TokenID, len(tokens))
	for index, token := range tokens {
		out, stepErr := oracle.Step(ctx, []SequenceBatchInput{{
			ID: 1, Tokens: []tokenizer.TokenID{token},
		}})
		if stepErr != nil {
			t.Fatalf("oracle step %d: %v", index, stepErr)
		}
		want[index] = greedyToken(t, out[0].Logits)
	}

	runner.mu.Lock()
	defer runner.mu.Unlock()
	release := func(cache *deviceKVCache) {
		if err := cache.Release(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	boundary, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, tokens[:2], nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer release(boundary)

	check := func(label string, cache *deviceKVCache) {
		t.Helper()
		if len(cache.SpanSelected) != 2 {
			t.Fatalf("%s selections = %v", label, cache.SpanSelected)
		}
		for index, selected := range cache.SpanSelected {
			if selected != want[2+index] {
				t.Fatalf(
					"%s position %d selected %d, oracle selected %d",
					label, index, selected, want[2+index],
				)
			}
		}
		if len(cache.SpanHidden.Data) != 2*int(runner.spec.EmbeddingLength) {
			t.Fatalf("%s hidden rows = %d values", label, len(cache.SpanHidden.Data))
		}
	}

	first, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, tokens[2:], boundary,
	)
	if err != nil {
		t.Fatal(err)
	}
	check("compiled span", first)
	session := first.session
	if session == nil {
		t.Fatal("span forward compiled no decode session")
	}
	if session.program.identity.tokenCount != 2 {
		t.Fatalf("span session token count = %d", session.program.identity.tokenCount)
	}
	release(first)

	second, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, tokens[2:], boundary,
	)
	if err != nil {
		t.Fatal(err)
	}
	check("replayed span", second)
	if second.session != session || session.replays == 0 {
		t.Fatalf(
			"span replay did not reuse the session: same=%t replays=%d",
			second.session == session, session.replays,
		)
	}
	release(second)
}
