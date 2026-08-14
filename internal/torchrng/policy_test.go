package torchrng

import "testing"

// TestTorchRNGPolicyGeometry pins the launch + Philox counter geometry for the
// golden 17/13 element fills against the adaptive_new reference math (device
// profile sm=128, max_threads_per_sm=1536, an Ada 4090-class device). This is
// model-free: it needs no GPU, so it keeps `plan -verify` meaningful in CI.
func TestTorchRNGPolicyGeometry(t *testing.T) {
	cases := []struct {
		elements          int64
		sm, threads       int
		wantGrid          int
		wantCounterOffset uint64
	}{
		{17, 128, 1536, 1, 4},
		{13, 128, 1536, 1, 4},
		{256, 128, 1536, 1, 4},
		// block=256, needed=ceil(1025/256)=5, grid=5, stride=256*5*4=5120,
		// counterOffset=ceil(1025/5120)*4=4.
		{1025, 128, 1536, 5, 4},
	}
	for _, c := range cases {
		grid, block, counterOffset, err := NullaryPolicy(c.elements, c.sm, c.threads)
		if err != nil {
			t.Fatalf("NullaryPolicy(%d): %v", c.elements, err)
		}
		if block != Block {
			t.Fatalf("elements=%d block=%d want %d", c.elements, block, Block)
		}
		if grid != c.wantGrid {
			t.Fatalf("elements=%d grid=%d want %d", c.elements, grid, c.wantGrid)
		}
		if counterOffset != c.wantCounterOffset {
			t.Fatalf("elements=%d counterOffset=%d want %d", c.elements, counterOffset, c.wantCounterOffset)
		}
	}
}

// TestTorchRNGPolicyGridCap verifies the grid saturates at smCount*blocksPerSM
// for a large extent (so the counter advance stays bounded by the launch, as in
// PyTorch's nullary launcher).
func TestTorchRNGPolicyGridCap(t *testing.T) {
	const sm, threads = 128, 1536
	grid, _, counterOffset, err := NullaryPolicy(10_000_000, sm, threads)
	if err != nil {
		t.Fatal(err)
	}
	maxGrid := sm * (threads / Block)
	if grid != maxGrid {
		t.Fatalf("grid=%d want capped %d", grid, maxGrid)
	}
	if counterOffset == 0 {
		t.Fatal("counterOffset must be positive for a multi-round launch")
	}
}

// TestTorchRNGPolicyRejectsBadInputs pins the error surface.
func TestTorchRNGPolicyRejectsBadInputs(t *testing.T) {
	if _, _, _, err := NullaryPolicy(0, 128, 1536); err == nil {
		t.Fatal("expected error for zero elements")
	}
	if _, _, _, err := NullaryPolicy(16, 0, 1536); err == nil {
		t.Fatal("expected error for zero sm")
	}
	if _, _, _, err := NullaryPolicy(16, 128, 128); err == nil {
		t.Fatal("expected error when max_threads_per_sm below block")
	}
}

// TestTorchRNGReserveAdvancesOffset checks that reserve returns the CURRENT
// offset as the draw point and advances by the policy counter step -- the
// mechanism that makes split fills continue one torch.randn sequence. Model-free.
func TestTorchRNGReserveAdvancesOffset(t *testing.T) {
	s := NewStream(31)
	if s.Offset() != 0 {
		t.Fatalf("fresh stream offset=%d want 0", s.Offset())
	}
	grid, drawAt, err := s.reserve(17, 128, 1536)
	if err != nil {
		t.Fatal(err)
	}
	if grid != 1 || drawAt != 0 {
		t.Fatalf("first reserve grid=%d drawAt=%d want 1,0", grid, drawAt)
	}
	if s.Offset() != 4 {
		t.Fatalf("offset after first=%d want 4", s.Offset())
	}
	_, drawAt2, err := s.reserve(13, 128, 1536)
	if err != nil {
		t.Fatal(err)
	}
	if drawAt2 != 4 {
		t.Fatalf("second drawAt=%d want 4 (continues the sequence)", drawAt2)
	}
	if s.Offset() != 8 {
		t.Fatalf("offset after second=%d want 8", s.Offset())
	}
}

// TestTorchRNGReserveOverflow ensures a near-max counter is rejected rather than
// silently wrapping (which would repeat the torch.randn stream).
func TestTorchRNGReserveOverflow(t *testing.T) {
	s := NewStream(1)
	s.offset = ^uint64(0) - 2 // advance of 4 cannot fit
	if _, _, err := s.reserve(17, 128, 1536); err == nil {
		t.Fatal("expected counter overflow error")
	}
}
