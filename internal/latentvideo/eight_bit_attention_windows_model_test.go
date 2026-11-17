//go:build windows && modeltest

package latentvideo

import (
	"fmt"
	"math"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// The 8-bit attention's frame bound (eight-bit-attention/wan-speed), declared before its first run: a clip drawn with
// 8-bit attention reads, over its frames as 8-bit pixels, a mean peak signal-to-noise ratio of at least 22 dB against
// the clip its request draws at full precision, a difference of detail and not of what is drawn, where an attention
// that computed wrongly draws noise far below it. A clip that equals the full-precision one is refused too: its
// attention was not the 8-bit kernel's.
const wanEightBitFramePSNR = 22.0

// eightBitPreset is the speed preset that asks 8-bit attention (profiles.json).
const eightBitPreset = "eight-bit-attention"

// TestWanEightBitAttentionKeepsItsClipsWithinTheirBound holds 8-bit attention, the eight-bit-attention speed preset, on
// Wan's 50-step path at the size a person draws (832 x 480, 81 frames): Wan2.1-T2V-1.3B loaded at full precision and
// again by the preset, which compiles every block's attention to the 8-bit kernel, draws the same two requests, and
// each 8-bit clip meets the frame bound against its full-precision one. Each clip's wall, its denoising and decoding,
// is logged beside the full-precision one's.
func TestWanEightBitAttentionKeepsItsClipsWithinTheirBound(t *testing.T) {
	cudatest.Require(t)
	fixture := newFrozenVideoFixture(t, false)
	prompts := []string{
		"a red fox trotting through fresh snow at dawn, its breath steaming in the cold air",
		"waves breaking against black volcanic rocks under a grey sky, spray drifting in the wind",
	}
	full := make([][][]uint8, len(prompts))
	walls := make([]time.Duration, len(prompts))
	for _, preset := range []string{"", eightBitPreset} {
		runtime := testcheck.Must(LoadWanRuntime(t.Context(), fixture.store, fixture.directory, fixture.program,
			WanRequest{Prompt: prompts[0], Preset: preset}))(t)
		if runtime.generator.eightBitAttention != (preset == eightBitPreset) {
			t.Fatalf("Wan loaded for preset %q computes its attention in 8 bits: %t", preset, runtime.generator.eightBitAttention)
		}
		for index, prompt := range prompts {
			request := testcheck.Must(runtime.resolve(t.Context(), WanRequest{Prompt: prompt, Seed: uint64(index + 1), Preset: preset}))(t)
			testcheck.NoError(t, runtime.admitsAttention(request))
			var ratios []float64
			changed := false
			started := time.Now()
			testcheck.NoError(t, runtime.draw(t.Context(), request, func(frame int, pixels []float32, _, _ int) error {
				shown := eightBitPixels(pixels)
				if preset == "" {
					full[index] = append(full[index], shown)
					return nil
				}
				if frame >= len(full[index]) {
					return fmt.Errorf("the 8-bit clip has a frame %d its full-precision clip of %d frames lacks", frame, len(full[index]))
				}
				ratio, differs := pixelPSNR(full[index][frame], shown)
				ratios, changed = append(ratios, ratio), changed || differs
				return nil
			}))
			wall := time.Since(started)
			if preset == "" {
				walls[index] = wall
				continue
			}
			mean, least := 0.0, math.Inf(1)
			for _, ratio := range ratios {
				mean += ratio / float64(len(ratios))
				least = min(least, ratio)
			}
			t.Logf("%q: %d frames, mean PSNR %.2f dB, least %.2f dB; drawn in %.1f s at full precision, %.1f s with 8-bit attention, %.2fx",
				prompt, len(ratios), mean, least, walls[index].Seconds(), wall.Seconds(), walls[index].Seconds()/wall.Seconds())
			switch {
			case len(ratios) != len(full[index]):
				t.Errorf("%q: the 8-bit clip has %d frames, its full-precision clip %d", prompt, len(ratios), len(full[index]))
			case !changed:
				t.Errorf("%q: the 8-bit clip equals its full-precision clip, so its attention was not the 8-bit kernel's", prompt)
			case mean < wanEightBitFramePSNR:
				t.Errorf("%q: the 8-bit clip reads a mean PSNR of %.2f dB against its full-precision clip, below the bound of %.0f dB",
					prompt, mean, wanEightBitFramePSNR)
			}
		}
		testcheck.NoError(t, runtime.Close(t.Context()))
	}
}

// eightBitPixels is a frame of signed unit values as the 8-bit pixels a clip shows.
func eightBitPixels(frame []float32) []uint8 {
	shown := make([]uint8, len(frame))
	for index, value := range frame {
		shown[index] = uint8(math.Round((max(-1, min(1, float64(value))) + 1) * 127.5))
	}
	return shown
}

// pixelPSNR is the peak signal-to-noise ratio of a frame's 8-bit pixels against a reference's, and whether any differ;
// identical frames read 99 dB.
func pixelPSNR(reference, frame []uint8) (float64, bool) {
	squared := 0.0
	for index := range reference {
		difference := float64(reference[index]) - float64(frame[index])
		squared += difference * difference
	}
	if squared == 0 {
		return 99, false
	}
	return 10 * math.Log10(255*255/(squared/float64(len(reference)))), true
}
