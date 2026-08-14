//go:build integration

package latentvideo

import (
	"math"
	"testing"
	"time"
)

// BenchmarkVAEDecodeProductionSpatialProbe: host wall at production
// 480x832 spatial extent over two latent frames (5 output frames), with the
// per-latent-frame extrapolation to the 21-frame clip logged verbatim.
func BenchmarkVAEDecodeProductionSpatialProbe(b *testing.B) {
	plan, checkpoint := compileRealVAEDecoderPlan(b)
	stats := vaeG0LatentStats(b)
	const latentFrames, latentH, latentW = 2, 60, 104
	z := make([]float32, plan.ZDim*latentFrames*latentH*latentW)
	for i := range z {
		z[i] = float32(math.Sin(float64(i)*0.61 + 0.17))
	}
	frames := 0
	var wall time.Duration
	for range b.N {
		frames = 0
		start := time.Now()
		if _, err := DecodeLatentVideo(checkpoint, plan, stats, z, latentFrames, latentH, latentW, func(index int, frame []float32, height, width int) error {
			frames++
			return nil
		}); err != nil {
			b.Fatal(err)
		}
		wall = time.Since(start)
	}
	// 21 latent frames = 1 + 5 four-frame chunks x 4; steady-state cost
	// scales with latent frames: extrapolate from the second latent frame
	// (first carries the weights load + first-chunk warmup).
	b.Logf("decode latent 2x%dx%d -> %d frames wall=%.2fs; naive 21-frame extrapolation=%.1fs",
		latentH, latentW, frames, wall.Seconds(), wall.Seconds()/2*21)
}
