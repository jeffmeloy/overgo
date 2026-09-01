package inference

import (
	"strings"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// TestEffectiveCachePositionEdges pins the position semantics prompt reuse
// and context shifting share: a nil cache holds no position, a v1 in-memory
// cache with a zero position field is append-only and reads its token
// count, and an explicit position wins over the token count.
func TestEffectiveCachePositionEdges(t *testing.T) {
	if position := effectiveCachePosition(nil); position != 0 {
		t.Fatalf("nil cache position = %d, want 0", position)
	}
	appendOnly := &KVCache{Tokens: 7}
	if position := effectiveCachePosition(appendOnly); position != 7 {
		t.Fatalf("append-only position = %d, want token count 7", position)
	}
	shifted := &KVCache{Tokens: 7, Position: 12}
	if position := effectiveCachePosition(shifted); position != 12 {
		t.Fatalf("explicit position = %d, want 12", position)
	}
}

// TestValidateStateValueRefusesShapeMismatch pins the cache-state schema
// gate: a state tensor whose data length disagrees with its shape refuses
// with both counts named, and a consistent one admits.
func TestValidateStateValueRefusesShapeMismatch(t *testing.T) {
	consistent := reference.Value{Shape: tensor.MustShape(2, 3), Data: make([]float32, 6)}
	if err := validateStateValue(consistent); err != nil {
		t.Fatalf("consistent state refused: %v", err)
	}
	mismatched := reference.Value{Shape: tensor.MustShape(2, 3), Data: make([]float32, 5)}
	if err := validateStateValue(mismatched); err == nil ||
		!strings.Contains(err.Error(), "5") || !strings.Contains(err.Error(), "6") {
		t.Fatalf("shape mismatch admitted: %v", err)
	}
}

// TestOwnsDevicePromptCache pins device-cache ownership resolution: nil
// caches are never owned, a registered device cache is, and a foreign
// device cache is not -- the trim and shift paths rely on this to refuse
// operating on caches the runner does not hold.
func TestOwnsDevicePromptCache(t *testing.T) {
	runner := &Runner{}
	if runner.ownsDevicePromptCache(nil) {
		t.Fatal("nil device cache owned")
	}
	owned := &deviceKVCache{}
	runner.promptCaches = append(runner.promptCaches, &cachedPrompt{Device: owned})
	if !runner.ownsDevicePromptCache(owned) {
		t.Fatal("registered device cache not owned")
	}
	if runner.ownsDevicePromptCache(&deviceKVCache{}) {
		t.Fatal("foreign device cache owned")
	}
}

// TestClearPromptCachesRefusals pins the cancellation-adjacent edges: a nil
// runner and a nil context both refuse before any cache detaches.
func TestClearPromptCachesRefusals(t *testing.T) {
	var missing *Runner
	if err := missing.ClearPromptCaches(t.Context()); err == nil {
		t.Fatal("nil runner cleared caches")
	}
	runner := &Runner{}
	if err := runner.ClearPromptCaches(nil); err == nil || !strings.Contains(err.Error(), "context") { //nolint:staticcheck
		t.Fatalf("nil context admitted: %v", err)
	}
}
