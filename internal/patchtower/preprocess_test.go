package patchtower

import "testing"

func TestSmartResizeAlignedRoundsToFactor(t *testing.T) {
	// 480x640 with factor 32 and generous pixel budget: round-to-even scaling.
	h, w, err := smartResizeAligned(480, 640, 32, 25088, 524288, 200)
	if err != nil {
		t.Fatal(err)
	}
	if h%32 != 0 || w%32 != 0 {
		t.Fatalf("resize %dx%d not factor-aligned", h, w)
	}
	if h*w > 524288 || h*w < 25088 {
		t.Fatalf("resize %dx%d outside pixel budget", h, w)
	}
	if _, _, err := smartResizeAligned(10, 4000, 32, 25088, 524288, 200); err == nil {
		t.Fatal("aspect ratio breach accepted")
	}
}

func TestPatchEmbedSourcePatchRoundTripsGrid(t *testing.T) {
	const gridH, gridW, merge = 4, 6, 2
	seen := make(map[int]bool)
	for token := 0; token < gridH*gridW; token++ {
		src, err := patchEmbedSourcePatch(token, gridH, gridW, merge)
		if err != nil {
			t.Fatal(err)
		}
		if src < 0 || src >= gridH*gridW || seen[src] {
			t.Fatalf("token %d -> src %d invalid or duplicate", token, src)
		}
		seen[src] = true
	}
}

func TestPatchifyMergedFramesShape(t *testing.T) {
	const rh, rw, patch, temporal, merge = 8, 8, 4, 1, 2
	frame := make([]float32, RGBChannels*rh*rw)
	for i := range frame {
		frame[i] = float32(i)
	}
	pv, gt, gh, gw, err := patchifyMergedFrames([][]float32{frame}, rh, rw, patch, temporal, merge)
	if err != nil {
		t.Fatal(err)
	}
	if gt != 1 || gh != 2 || gw != 2 {
		t.Fatalf("grid [%d,%d,%d]", gt, gh, gw)
	}
	if len(pv) != gt*gh*gw*RGBChannels*temporal*patch*patch {
		t.Fatalf("pixel values len %d", len(pv))
	}
}
