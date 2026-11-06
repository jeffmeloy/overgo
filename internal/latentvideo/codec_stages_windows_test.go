//go:build windows

package latentvideo

import (
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// The small clip the test generates: its frames and side, its steps, and the trajectory's shift and guidance.
const (
	stageClipFrames = 5
	stageClipSide   = 64
	stageClipSteps  = 2
	stageClipShift  = 3
	stageClipGuide  = 5
	stageClipScale  = 1e-3
)

// TestAVideoStageLeavesNoScratchForTheNext: a Wan request gives back each stage's memory before the next stage grows its
// own: the denoiser's run memory before the autoencoder decodes, and the autoencoder's workspace once the clip is decoded,
// so between requests the denoiser holds no arena or idle buffer and the autoencoder no workspace. A second request makes
// the first's frames, bit for bit, after its stages grow their memory again.
func TestAVideoStageLeavesNoScratchForTheNext(t *testing.T) {
	cudatest.Require(t)
	dir := wanModelDir(t)
	profile := testcheck.Must(ResolveProfile(dir))(t)
	generator := testcheck.Must(NewGenerator(GeneratorConfig{
		ModelDirectory: dir, Policy: profile.Policy, LatentStats: profile.LatentStats,
		Frames: stageClipFrames, Width: stageClipSide, Height: stageClipSide,
	}))(t)
	defer generator.Close()
	condition, initial := make([]float32, generator.ContextElements()), make([]float32, generator.Geometry().Elements())
	for index := range condition {
		condition[index] = float32(math.Sin(float64(index))) * stageClipScale
	}
	for index := range initial {
		initial[index] = float32(math.Cos(float64(index)))
	}
	clip := func() []float32 {
		t.Helper()
		var frames []float32
		if _, err := generator.Generate(t.Context(), GenerateRequest{
			Steps: stageClipSteps, Shift: stageClipShift, GuideScale: stageClipGuide,
			CondContext: condition, UncondContext: make([]float32, len(condition)), InitialSample: initial,
			Sink: func(_ int, frame []float32, _, _ int) error { frames = append(frames, frame...); return nil },
		}); err != nil {
			t.Fatal(err)
		}
		metrics := testcheck.Must(generator.denoiser.cuda.Metrics(t.Context()))(t)
		if metrics.ArenaCommittedBytes != 0 || metrics.PoolRetainedBytes != 0 || len(generator.decoder.buffers) != 0 {
			t.Fatalf("between requests the denoiser holds a %d-byte arena and %d idle bytes, and the autoencoder %d workspaces",
				metrics.ArenaCommittedBytes, metrics.PoolRetainedBytes, len(generator.decoder.buffers))
		}
		return frames
	}
	first, second := clip(), clip()
	if len(first) == 0 || !slices.EqualFunc(first, second, func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }) {
		t.Fatalf("a second request made other frames: %d values against %d", len(second), len(first))
	}
}
