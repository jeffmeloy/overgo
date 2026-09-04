//go:build windows

package inference

import (
	"testing"

	"overgo/internal/tokenizer"
)

// TestHermeticCUDAContinuationScoringHoldsDeviceMemory: scoring one case
// after another must not grow device memory: every buffer a case takes
// (prompt cache pages, retained outputs, logits) returns to the pool
// before the next case, so twenty scorings after warm-up leave the
// device's used bytes where two left them. The MuSR pass on the E4B
// grew about a gigabyte per case until the 49 GB device filled and the
// driver paged, with identical GEMMs running two hundred times slower.
func TestHermeticCUDAContinuationScoringHoldsDeviceMemory(t *testing.T) {
	runner := openHermeticScoringRunner(t)
	prompt := []tokenizer.TokenID{1, 4, 5, 6}
	continuations := [][]tokenizer.TokenID{{6}, {7, 4}, {5, 6, 7}}
	score := func() {
		runner.mu.Lock()
		defer runner.mu.Unlock()
		if _, err := runner.scoreContinuationsDeviceLocked(t.Context(), prompt, continuations); err != nil {
			t.Fatal(err)
		}
	}
	for range 2 {
		score()
	}
	warm, err := runner.DeviceMemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		score()
	}
	after, err := runner.DeviceMemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if after.CurrentBytes > warm.CurrentBytes {
		t.Fatalf("device memory grew by %d bytes over twenty scorings (%d -> %d)", after.CurrentBytes-warm.CurrentBytes, warm.CurrentBytes, after.CurrentBytes)
	}
}
