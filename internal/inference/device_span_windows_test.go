//go:build windows

package inference

import (
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tokenizer"
)

func greedyToken(t *testing.T, logits []float32) tokenizer.TokenID {
	t.Helper()
	if len(logits) == 0 {
		t.Fatal("greedy oracle logits are empty")
	}
	best := 0
	for index, value := range logits {
		if value > logits[best] {
			best = index
		}
	}
	return tokenizer.TokenID(best)
}

// TestDeviceDecodeSpan proves the MTP verify batch: a bounded token span
// appended in one device forward yields one greedy selection per consumed
// position, matching a teacher-forced host decode (the greedy device step
// self-feeds its retained selection, so the host path is the oracle that
// consumes exactly the given tokens).
func TestDeviceDecodeSpan(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := writeHermeticLlamaGGUF(t)
	runner, err := openF32FixtureRunner(path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	tokens := []tokenizer.TokenID{1, 4, 5, 6}

	// Teacher-forced host oracle: consume each span token one step at a
	// time and record the greedy selection after every position.
	oracle, err := runner.NewContinuousBatch(ContinuousBatchOptions{MaxSequences: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer oracle.Close(context.Background())
	want := make([]tokenizer.TokenID, len(tokens))
	var oracleTokens, oraclePosition uint32
	for index, token := range tokens {
		out, stepErr := oracle.Step(ctx, []SequenceBatchInput{{
			ID: 1, Tokens: []tokenizer.TokenID{token},
		}})
		if stepErr != nil {
			t.Fatalf("oracle step %d: %v", index, stepErr)
		}
		want[index] = greedyToken(t, out[0].Logits)
		oracleTokens, oraclePosition = out[0].Tokens, out[0].Position
	}

	runner.mu.Lock()
	defer runner.mu.Unlock()

	// Whole-history span from an empty cache.
	span, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedySpan, tokens, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := span.Release(context.Background()); err != nil {
			t.Fatal(err)
		}
	}()
	if len(span.SpanSelected) != len(tokens) {
		t.Fatalf("span selections = %d, want %d", len(span.SpanSelected), len(tokens))
	}
	for index := range tokens {
		if span.SpanSelected[index] != want[index] {
			t.Fatalf(
				"span position %d selected %d, teacher-forced oracle selected %d",
				index, span.SpanSelected[index], want[index],
			)
		}
	}
	if span.Tokens != oracleTokens || span.Position != oraclePosition {
		t.Fatalf(
			"span cache tokens/position = %d/%d, oracle = %d/%d",
			span.Tokens, span.Position, oracleTokens, oraclePosition,
		)
	}

	// The MTP shape proper: a committed prefix cache plus a draft span.
	prefix, err := runner.forwardDeviceCachedGreedyStepLocked(ctx, tokens[:1], nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := prefix.Release(context.Background()); err != nil {
			t.Fatal(err)
		}
	}()
	draft, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedySpan, tokens[1:], prefix)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := draft.Release(context.Background()); err != nil {
			t.Fatal(err)
		}
	}()
	if len(draft.SpanSelected) != len(tokens)-1 {
		t.Fatalf("draft selections = %d, want %d", len(draft.SpanSelected), len(tokens)-1)
	}
	for index := range draft.SpanSelected {
		if draft.SpanSelected[index] != want[index+1] {
			t.Fatalf(
				"draft position %d selected %d, teacher-forced oracle selected %d",
				index, draft.SpanSelected[index], want[index+1],
			)
		}
	}
	if draft.Tokens != oracleTokens || draft.Position != oraclePosition {
		t.Fatalf(
			"draft cache tokens/position = %d/%d, oracle = %d/%d",
			draft.Tokens, draft.Position, oracleTokens, oraclePosition,
		)
	}
}
