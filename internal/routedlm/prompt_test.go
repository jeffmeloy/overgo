package routedlm

import "testing"

func TestVisualSegmentsWidensImageSpans(t *testing.T) {
	// Text runs of >=2 zeros are separators; the spans between them (the
	// image spans plus their bordering singleton rows) become bidirectional
	// windows widened one row past their end. Rows inside a separator fall
	// outside every segment and attend causally.
	mask := []int{0, 0, 1, 1, 1, 0, 0}
	got := VisualSegments(mask)
	want := [][2]int{{2, 6}}
	if len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("segments %v, want %v", got, want)
	}
	// Leading text before the first image span keeps a widened window.
	got = VisualSegments([]int{1, 1, 0, 0, 1, 1})
	want = [][2]int{{0, 3}, {4, 6}}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("segments %v, want %v", got, want)
	}
}

func TestSegmentRangeCausalOutsideSegments(t *testing.T) {
	segments := [][2]int{{0, 3}}
	start, end := segmentRange(segments, 5, 6)
	if start != 0 || end != 6 {
		t.Fatalf("range [%d,%d), want [0,6)", start, end)
	}
	start, end = segmentRange(segments, 1, 6)
	if start != 0 || end != 3 {
		t.Fatalf("range [%d,%d), want [0,3)", start, end)
	}
}

func TestRopeInvFreqBaseDynamicAlpha(t *testing.T) {
	cfg := Config{HeadDim: 128, RopeTheta: 10000, RopeScaling: map[string]any{"type": "dynamic", "alpha": 1000.0}}
	base, err := RopeInvFreqBase(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// theta * alpha^(d/(d-2)) with d=128: 1e4 * 1000^(128/126).
	if base <= 1e7 || base >= 1.2e7 {
		t.Fatalf("dynamic base = %g", base)
	}
	cfg.RopeScaling = nil
	base, err = RopeInvFreqBase(cfg)
	if err != nil || base != 10000 {
		t.Fatalf("plain base = %g err=%v", base, err)
	}
}

func TestModalityMaskAndPositions(t *testing.T) {
	specials := PromptSpecials{Image: 7, Video: 8}
	ids := []int{1, 7, 7, 8, 2}
	positions := ImageMaskPositions(ids, specials)
	if len(positions) != 3 || positions[0] != 1 || positions[2] != 3 {
		t.Fatalf("positions = %v", positions)
	}
	mask := ModalityMask(len(ids), positions)
	want := []int{0, 1, 1, 1, 0}
	for i := range want {
		if mask[i] != want[i] {
			t.Fatalf("mask = %v, want %v", mask, want)
		}
	}
}
