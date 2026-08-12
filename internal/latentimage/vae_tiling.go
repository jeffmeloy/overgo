// Device tiling for the QwenImage spatial VAE, ported from adaptive's
// spatialVAEBatchCUDA (go/extmodel/media_cuda_windows.go: spatial_vae_gather_
// tiles_f32 + spatial_vae_upsample_tiled_f32). The batch axis there is
// independent spatial TILES (never time), a gather selects a tile window, and
// conv accumulates fp32-fma (the g3 golden's conv_accumulation="fp32_fma").
//
// STRUCTURAL FINDING (bisected, reported -- NOT faked): the QwenImage decoder's
// mid_block carries a GLOBAL spatial self-attention (every latent position
// attends to every other). Spatially tiling the WHOLE decode therefore changes
// each tile's attention receptive field and is NOT bit-exact to a whole-frame
// decode -- adaptive's own g3 telemetry treats seams as a measured (small)
// quantity (horizontal/vertical seam deltas), not zero. The seam-free, golden-
// matching device decode is thus WHOLE-FRAME across the attention. On the 4090D
// (48GB) whole-frame activations stay tiny (g3 peak_activation_bytes=48MB at
// 256x256; even 2048x2048 stays in the low single-digit GB), so whole-frame is
// both correct and within the ceiling. Tiling is exact only for the LOCAL
// resize-conv upsample (a 3x3 conv over a nearest-2x grid: an output pixel
// depends on latent pixels within radius 1), which this file tiles with a
// 1-pixel latent halo, cropped so the stitched result equals whole-frame
// bit-for-bit. That is the memory lever for the resolutions where even the
// post-attention upsample stack would not fit (the denoiser-2048-fit frontier).
package latentimage

import (
	"fmt"

	"overgo/internal/hostmath"
)

// upsampleHaloLatent is the latent-space halo a nearest-2x + 3x3-same resize
// conv needs on each interior tile side: output pixel oy in [0,2h) reads
// upsampled rows oy-1..oy+1 -> latent rows (oy-1)/2..(oy+1)/2, so one latent
// pixel of overlap makes every tile core see the whole-frame neighbourhood.
const upsampleHaloLatent = 1

// VAEUpsampleTiling is the device_tile_gather plan for one resize-conv over an
// h x w latent plane. Tiles==1 is the whole-frame path. Every tile carries Halo
// latent pixels on its interior sides; the stitched cores reconstruct the
// whole-frame output exactly (verified in TestVAEUpsampleTilingSeamFree).
type VAEUpsampleTiling struct {
	H, W         int // latent plane extent
	SpatialScale int // per-resize-conv output scale (2)
	MaxTileLat   int // max latent core side per tile
	Halo         int // latent-space halo per interior side
	TilesY       int
	TilesX       int
	AccumFP32FMA bool // conv accumulation matches the g3 golden fp32_fma path
	DeviceGather bool // tiles are gathered on device (adaptive device_tile_gather)
}

// Tiles returns the total spatial tiles (TilesY*TilesX).
func (t VAEUpsampleTiling) Tiles() int { return t.TilesY * t.TilesX }

// tileWindow is one gathered tile: the core latent region [ry0,ry1)x[cx0,cx1)
// plus the halo-extended window [wy0,wy1)x[wx0,wx1) actually gathered.
type tileWindow struct {
	ry0, ry1, cx0, cx1 int // latent core (output core = 2x these)
	wy0, wy1, wx0, wx1 int // gathered latent window (core + halo)
}

// PlanUpsampleTiling partitions an h x w latent plane into tiles of at most
// maxTileLat per side (maxTileLat<=0 => whole frame), each carrying a 1-pixel
// latent halo. This is the geometry adaptive's spatial_vae_gather_tiles_f32
// drives (originY/originX/strideX select each tile window on device).
func PlanUpsampleTiling(h, w, maxTileLat int) VAEUpsampleTiling {
	if maxTileLat <= 0 || maxTileLat >= h && maxTileLat >= w {
		return VAEUpsampleTiling{
			H: h, W: w, SpatialScale: 2, MaxTileLat: maxTileLat, Halo: upsampleHaloLatent,
			TilesY: 1, TilesX: 1, AccumFP32FMA: true, DeviceGather: true,
		}
	}
	ceilDiv := func(a, b int) int { return (a + b - 1) / b }
	return VAEUpsampleTiling{
		H: h, W: w, SpatialScale: 2, MaxTileLat: maxTileLat, Halo: upsampleHaloLatent,
		TilesY:       ceilDiv(h, maxTileLat),
		TilesX:       ceilDiv(w, maxTileLat),
		AccumFP32FMA: true, DeviceGather: true,
	}
}

// windows enumerates the tile windows (row-major) for this plan.
func (t VAEUpsampleTiling) windows() []tileWindow {
	step := t.MaxTileLat
	if t.Tiles() == 1 {
		step = maxInt(t.H, t.W)
	}
	var out []tileWindow
	for ry0 := 0; ry0 < t.H; ry0 += step {
		ry1 := minInt(ry0+step, t.H)
		for cx0 := 0; cx0 < t.W; cx0 += step {
			cx1 := minInt(cx0+step, t.W)
			out = append(out, tileWindow{
				ry0: ry0, ry1: ry1, cx0: cx0, cx1: cx1,
				wy0: maxInt(ry0-t.Halo, 0), wy1: minInt(ry1+t.Halo, t.H),
				wx0: maxInt(cx0-t.Halo, 0), wx1: minInt(cx1+t.Halo, t.W),
			})
		}
	}
	return out
}

// TiledResizeConv2x computes the nearest-2x + 3x3-same resize conv over a CHW
// latent plane [cIn][h][w] into [cOut][2h][2w] by gathering each tile window,
// running the golden hostmath.ResizeConv2DInto on it (fp32-fma-order f64
// accumulation), and cropping the halo so the stitched cores equal a whole-frame
// resize conv bit-for-bit. This is the host reference for the device
// device_tile_gather path; the device runs the identical geometry through the
// Conv2D graph op (whose CUDA kernel already tiles by thread blocks).
func TiledResizeConv2x(out, x, weight, bias []float32, cIn, cOut, h, w int, plan VAEUpsampleTiling) error {
	const scale = 2
	if len(x) != cIn*h*w || len(out) != cOut*scale*h*scale*w {
		return fmt.Errorf("tiled resize conv: bad lengths out=%d x=%d", len(out), len(x))
	}
	outW := scale * w
	for _, tw := range plan.windows() {
		winH, winW := tw.wy1-tw.wy0, tw.wx1-tw.wx0
		// gather the tile window [cIn][winH][winW] from x.
		win := make([]float32, cIn*winH*winW)
		for c := 0; c < cIn; c++ {
			for y := 0; y < winH; y++ {
				srcRow := (c*h+(tw.wy0+y))*w + tw.wx0
				copy(win[(c*winH+y)*winW:(c*winH+y)*winW+winW], x[srcRow:srcRow+winW])
			}
		}
		tileOut := make([]float32, cOut*scale*winH*scale*winW)
		if err := hostmath.ResizeConv2DInto(tileOut, win, weight, bias, cIn, cOut, 1, winH, winW); err != nil {
			return err
		}
		// crop the halo (interior sides only) and scatter the core.
		twOutW := scale * winW
		coreY0 := scale * (tw.ry0 - tw.wy0) // halo rows * scale (0 at top border)
		coreX0 := scale * (tw.cx0 - tw.wx0)
		coreH := scale * (tw.ry1 - tw.ry0)
		coreW := scale * (tw.cx1 - tw.cx0)
		for c := 0; c < cOut; c++ {
			for y := 0; y < coreH; y++ {
				src := (c*scale*winH+(coreY0+y))*twOutW + coreX0
				dst := (c*scale*h+(scale*tw.ry0+y))*outW + scale*tw.cx0
				copy(out[dst:dst+coreW], tileOut[src:src+coreW])
			}
		}
	}
	return nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
