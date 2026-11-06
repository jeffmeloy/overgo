//go:build windows

package inference

import (
	"slices"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modeltest"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

// The requests the test serves: a short prompt, a long one and the tokens each decodes.
const (
	shortRead     = 16
	longRead      = 16384
	readDecodes   = 8
	readTokenBase = 1000
	readTokenSpan = 5000
)

// TestALongReadLeavesADenseRunnerItsDecodeScratch: a dense model's runner frees the arena a long prompt's read grew,
// once the prompt is read, where it exceeds what the decode needs by more than the executor holds idle, and the decode
// takes its own again: after a 16384-token request on Qwen3.5-9B the arena is no larger than a short request's, and the
// card holds less than a short request left it and the read's arena, where the read's 1.6 GB stayed for the runner's
// life. A short request's read leaves the arena and the graphs captured over it as they are, so a short request after
// another captures nothing again. Every request decodes the tokens it decodes alone.
func TestALongReadLeavesADenseRunnerItsDecodeScratch(t *testing.T) {
	cudatest.Require(t)
	runner := testcheck.Must(openNativeFixtureRunner(modeltest.Directory(t, "Qwen3.5-9B-Q8_0.gguf"), OpenOptions{DeviceOrdinal: 0}))(t)
	defer runner.Close()
	generate := func(tokens int) ([]tokenizer.TokenID, executor.ExecutionMetrics, uint64) {
		t.Helper()
		ids := make([]tokenizer.TokenID, tokens)
		for i := range ids {
			ids[i] = tokenizer.TokenID(readTokenBase + i%readTokenSpan)
		}
		sampler := testcheck.Must(sampling.New(sampling.Config{}))(t)
		out, _, err := runner.Generate(t.Context(), "", GenerateOptions{MaxNewTokens: readDecodes, Sampler: sampler, PromptTokenIDs: ids})
		testcheck.NoError(t, err)
		metrics := testcheck.Must(runner.cuda.Metrics(t.Context()))(t)
		return out[len(out)-readDecodes:], metrics, driver.ProcessCurrentBytes()
	}
	short, first, firstHeld := generate(shortRead)
	repeated, second, _ := generate(shortRead)
	long, afterLong, longHeld := generate(longRead)
	again, _, _ := generate(shortRead)
	alone, _, _ := generate(longRead)
	t.Logf("the arena is %d bytes after a short request, %d at the long read's peak, %d after it; the card held %d bytes after a short request and %d after the long one",
		first.ArenaCommittedBytes, afterLong.ArenaPeakRequiredBytes, afterLong.ArenaCommittedBytes, firstHeld, longHeld)
	if second.GraphInstantiations != first.GraphInstantiations || second.ArenaCommittedBytes != first.ArenaCommittedBytes {
		t.Fatalf("a short request after another took an arena of %d bytes and %d graph instantiations, want the first's %d and %d",
			second.ArenaCommittedBytes, second.GraphInstantiations, first.ArenaCommittedBytes, first.GraphInstantiations)
	}
	if afterLong.ArenaCommittedBytes > first.ArenaCommittedBytes || longHeld >= firstHeld+afterLong.ArenaPeakRequiredBytes {
		t.Fatalf("after a long request the arena is %d bytes and the card holds %d, want at most the short request's %d and less than its %d and the read's %d",
			afterLong.ArenaCommittedBytes, longHeld, first.ArenaCommittedBytes, firstHeld, afterLong.ArenaPeakRequiredBytes)
	}
	if !slices.Equal(repeated, short) || !slices.Equal(again, short) || !slices.Equal(alone, long) {
		t.Fatalf("a request decoded other tokens beside a long read: short %v, %v, %v, long %v then %v", short, repeated, again, long, alone)
	}
}
