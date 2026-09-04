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
	// Every prompt length the fixture's context admits, with single- and
	// multi-token candidates: a pass over a suite sees a new length on
	// nearly every case, and a candidate past its first token takes the
	// per-token decode step.
	tokens := []tokenizer.TokenID{1, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5}
	continuations := [][]tokenizer.TokenID{{6}, {7, 4}, {5, 6, 7}}
	score := func(length int) {
		runner.mu.Lock()
		defer runner.mu.Unlock()
		if _, err := runner.scoreContinuationsDeviceLocked(t.Context(), tokens[:length], continuations); err != nil {
			t.Fatal(err)
		}
	}
	cycle := func() {
		for length := 2; length <= len(tokens); length++ {
			score(length)
		}
	}
	cycle()
	warm, err := runner.DeviceMemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for range 4 {
		cycle()
	}
	after, err := runner.DeviceMemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if after.CurrentBytes > warm.CurrentBytes || after.Allocations > warm.Allocations {
		t.Fatalf("device memory grew over four cycles of every prompt length: %d -> %d bytes, %d -> %d allocations",
			warm.CurrentBytes, after.CurrentBytes, warm.Allocations, after.Allocations)
	}
}
