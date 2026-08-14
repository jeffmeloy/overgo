package latentimage

import (
	"testing"

	"overgo/internal/hostmath"
)

// TestVAEUpsampleTilingSeamFree proves the ported device_tile_gather geometry is
// bit-for-bit identical to a whole-frame resize conv: partitioning the latent
// into tiles (each with a 1-pixel halo) and stitching the cropped cores matches
// hostmath.ResizeConv2DInto over the whole plane exactly, at every tile count.
// This is the seam-free guarantee for the LOCAL upsample (the only exact tiling
// point -- see the mid-block global-attention finding in vae_tiling.go).
func TestVAEUpsampleTilingSeamFree(t *testing.T) {
	const cIn, cOut, h, w = 5, 4, 9, 7
	x := make([]float32, cIn*h*w)
	weight := make([]float32, cOut*cIn*9)
	bias := make([]float32, cOut)
	state := uint64(0xabcdef)
	fill := func(s []float32, scale float64) {
		for i := range s {
			state = state*6364136223846793005 + 1442695040888963407
			s[i] = float32((float64(state>>11)/float64(1<<53) - 0.5) * scale)
		}
	}
	fill(x, 2.0)
	fill(weight, 0.4)
	fill(bias, 0.2)

	whole := make([]float32, cOut*2*h*2*w)
	if err := hostmath.ResizeConv2DInto(whole, x, weight, bias, cIn, cOut, 1, h, w); err != nil {
		t.Fatalf("whole-frame resize conv: %v", err)
	}

	for _, maxTile := range []int{2, 3, 4, 100} {
		plan := PlanUpsampleTiling(h, w, maxTile)
		tiled := make([]float32, cOut*2*h*2*w)
		if err := TiledResizeConv2x(tiled, x, weight, bias, cIn, cOut, h, w, plan); err != nil {
			t.Fatalf("tiled resize conv (maxTile=%d): %v", maxTile, err)
		}
		diff := 0
		for i := range whole {
			if whole[i] != tiled[i] {
				diff++
			}
		}
		if diff != 0 {
			t.Fatalf("maxTile=%d tiles=%dx%d: %d/%d pixels differ from whole-frame (seam!)",
				maxTile, plan.TilesY, plan.TilesX, diff, len(whole))
		}
		t.Logf("maxTile=%d tiles=%dx%d halo=%d: bit-exact vs whole-frame (%d pixels)",
			maxTile, plan.TilesY, plan.TilesX, plan.Halo, len(whole))
	}
}

// TestPlanUpsampleTilingGeometry checks the plan invariants: whole-frame when
// the tile bound covers the plane, and correct ceil-div tile counts otherwise.
func TestPlanUpsampleTilingGeometry(t *testing.T) {
	if p := PlanUpsampleTiling(32, 32, 0); p.Tiles() != 1 || !p.DeviceGather || !p.AccumFP32FMA {
		t.Fatalf("whole-frame plan wrong: %+v", p)
	}
	if p := PlanUpsampleTiling(32, 32, 64); p.Tiles() != 1 {
		t.Fatalf("tile bound above extent must be whole-frame: %+v", p)
	}
	p := PlanUpsampleTiling(32, 24, 10)
	if p.TilesY != 4 || p.TilesX != 3 || p.Halo != upsampleHaloLatent {
		t.Fatalf("ceil-div tiling wrong: %+v", p)
	}
}
