package latentvideo

import (
	"math"
	"os"
	"testing"
	"time"
)

// TestVAEDecodeProductionSpatialProbe: host decode wall at the production
// 480x832 spatial extent over two latent frames (5 output frames), with the
// per-latent-frame extrapolation to the 21-frame clip logged verbatim.
// Gated: set OVERGO_WAN_BENCH=1 (CPU-heavy measurement, not a parity test).
func TestVAEDecodeProductionSpatialProbe(t *testing.T) {
	if os.Getenv("OVERGO_WAN_BENCH") == "" {
		t.Skip("set OVERGO_WAN_BENCH=1 to run the host VAE decode probe")
	}
	plan, checkpoint := compileRealVAEDecoderPlan(t)
	stats := vaeG0LatentStats(t)
	const latentFrames, latentH, latentW = 2, 60, 104
	z := make([]float32, plan.ZDim*latentFrames*latentH*latentW)
	for i := range z {
		z[i] = float32(math.Sin(float64(i)*0.61 + 0.17))
	}
	frames := 0
	start := time.Now()
	if _, err := DecodeLatentVideo(checkpoint, plan, stats, z, latentFrames, latentH, latentW, func(index int, frame []float32, height, width int) error {
		frames++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	wall := time.Since(start)
	// 21 latent frames = 1 + 5 four-frame chunks x 4; steady-state cost
	// scales with latent frames: extrapolate from the second latent frame
	// (first carries the weights load + first-chunk warmup).
	t.Logf("decode latent 2x%dx%d -> %d frames wall=%.2fs; naive 21-frame extrapolation=%.1fs",
		latentH, latentW, frames, wall.Seconds(), wall.Seconds()/2*21)
}
