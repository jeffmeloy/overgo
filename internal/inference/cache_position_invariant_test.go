package inference

import (
	"testing"

	"overgo/internal/model"
)

// TestCachePositionInvariantHonorsSectionRope pins the multimodal
// position contract: sequential rope refuses a next position below the
// active token count, while a declared multi-section rotary embedding
// accepts it — vision tokens advance positions by grid extent, so a
// prompt's next position legitimately trails its KV entries.
func TestCachePositionInvariantHonorsSectionRope(t *testing.T) {
	var sequential model.Spec
	if err := cachePositionInvariant(sequential, 334, 334); err != nil {
		t.Fatalf("sequential full position refused: %v", err)
	}
	if err := cachePositionInvariant(sequential, 54, 334); err == nil {
		t.Fatal("sequential trailing position accepted; that state is corruption")
	}
	var sectioned model.Spec
	sectioned.RopeSections = [4]int32{24, 20, 20, 0}
	if err := cachePositionInvariant(sectioned, 54, 334); err != nil {
		t.Fatalf("section-rope trailing position refused: %v", err)
	}
	if err := cachePositionInvariant(sectioned, 0, 334); err == nil {
		t.Fatal("section-rope zero position accepted with active tokens")
	}
}
