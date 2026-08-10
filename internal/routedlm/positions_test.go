package routedlm

import "testing"

// Mirrors the adaptive sensenova_edit_test.go case shapes: a 3x2 image block
// (token width 3) embedded in text — the block shares one time index,
// rasterizes (H, W) row-major, and attends bidirectionally within itself
// while text stays causal.
func TestBlockPositionsAndCausalWindowsImageBlock(t *testing.T) {
	//                 text     image block (3x2)  text
	mask := []int{0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0}
	tokenWidth := 3
	positions, err := BlockPositions(mask, tokenWidth)
	if err != nil {
		t.Fatal(err)
	}
	sourceRows := []int{3, 4, 5, 6, 7, 8}
	blockTime := positions[sourceRows[0]].Time
	if blockTime != 3 {
		t.Fatalf("block time %d, want 3", blockTime)
	}
	for index, row := range sourceRows {
		p := positions[row]
		if p.Branch != 1 || p.Time != blockTime || p.H != index/tokenWidth || p.W != index%tokenWidth {
			t.Fatalf("source row %d position %+v (index %d)", row, p, index)
		}
	}
	wantTextTimes := map[int]int{0: 0, 1: 1, 2: 2, 9: 4, 10: 5}
	for row, want := range wantTextTimes {
		p := positions[row]
		if p.Branch != 0 || p.Time != want || p.H != 0 || p.W != 0 {
			t.Fatalf("text row %d position %+v, want time %d", row, p, want)
		}
	}
	times := make([]int, len(positions))
	for row, p := range positions {
		times[row] = p.Time
	}
	ends, err := BlockCausalEnds(times)
	if err != nil {
		t.Fatal(err)
	}
	// First source row sees through the last source row (adaptive assertion).
	if ends[sourceRows[0]] != int32(sourceRows[len(sourceRows)-1]) {
		t.Fatalf("block causal ends = %v", ends)
	}
	windows, err := BlockCausalWindows(positions)
	if err != nil {
		t.Fatal(err)
	}
	wantEnds := []int32{0, 1, 2, 8, 8, 8, 8, 8, 8, 9, 10}
	for row, want := range wantEnds {
		if ends[row] != want {
			t.Fatalf("row %d end %d, want %d (ends %v)", row, ends[row], want, ends)
		}
		if windows[row] != [2]int{0, int(want) + 1} {
			t.Fatalf("row %d window %v, want [0,%d)", row, windows[row], want+1)
		}
	}
}

func TestBlockPositionsTwoBlocksAdvanceTime(t *testing.T) {
	mask := []int{0, 1, 1, 0, 1, 1}
	positions, err := BlockPositions(mask, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantTimes := []int{0, 1, 1, 2, 3, 3}
	for row, want := range wantTimes {
		if positions[row].Time != want {
			t.Fatalf("row %d time %d, want %d", row, positions[row].Time, want)
		}
	}
	if positions[4].H != 0 || positions[4].W != 0 || positions[5].W != 1 {
		t.Fatalf("second block restarts raster: %+v %+v", positions[4], positions[5])
	}
}

func TestBlockCausalEndsRejectsInvalidTimes(t *testing.T) {
	if _, err := BlockCausalEnds(nil); err == nil {
		t.Fatal("empty times accepted")
	}
	if _, err := BlockCausalEnds([]int{0, -1}); err == nil {
		t.Fatal("negative time accepted")
	}
	if _, err := BlockCausalEnds([]int{2, 2, 1}); err == nil {
		t.Fatal("decreasing times accepted")
	}
}

// SegmentWindows must reproduce segmentRange row-by-row (the rxbrain rule).
func TestSegmentWindowsMatchSegmentRange(t *testing.T) {
	mask := []int{1, 1, 0, 0, 1, 1, 1}
	segments := VisualSegments(mask)
	windows := SegmentWindows(segments, len(mask))
	for row := 0; row < len(mask); row++ {
		start, end := segmentRange(segments, row, len(mask))
		if windows[row] != [2]int{start, end} {
			t.Fatalf("row %d window %v, want [%d,%d)", row, windows[row], start, end)
		}
	}
}
