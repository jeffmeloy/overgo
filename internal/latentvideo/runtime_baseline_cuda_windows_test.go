//go:build windows && integration

package latentvideo

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const (
	wanPythonProductionWall = 463.4
	wanOvergoDenoiseCeiling = 370.0
	wanOvergoDecodeCeiling  = 60.0
	wanOvergoPeakCeiling    = uint64(7_880_000_000)
)

func productionNoiseGrid(t testing.TB, elements int) int {
	t.Helper()
	library := testcheck.Must(driver.Open())(t)
	defer library.Close()
	testcheck.NoError(t, library.Init())
	info := testcheck.Must(library.DeviceInfo(0))(t)
	grid := (elements + 1023) / 1024
	if ceiling := info.MultiprocessorCount * 6; grid > ceiling {
		grid = ceiling
	}
	return max(grid, 1)
}

// BenchmarkWanRealArtifactLeadership: production Wan 1.3B denoise ratchet.
// Run once: -run '^$' -bench '^BenchmarkWanRealArtifactLeadership$' -benchtime=1x.
func BenchmarkWanRealArtifactLeadership(b *testing.B) {
	cudatest.Require(b)

	config, weights := denoiserFixture(b)
	geometry := testcheck.Must(config.CompileLatentGeometry(81, 832, 480))(b)
	program := testcheck.Must(CompileDenoiserProgramPrecision(config, weights, geometry, DenoiserPrecision{
		MatmulWeights: dtype.BF16, RoundAttentionStorage: true,
	}))(b)
	fixtureDir := testutil.FixturePath(b, "wan")
	contextElements := config.TextLen * config.Dim
	cond := loadRawContext(b, fixtureDir, "raw/g1_cond_context.f32le", contextElements)
	uncond := loadRawContext(b, fixtureDir, "raw/g1_uncond_context.f32le", contextElements)

	b.ResetTimer()
	for b.Loop() {
		started := time.Now()
		denoiser := testcheck.Must(NewDenoiserCUDASession(program, 0))(b)
		result, err := denoiser.Denoise(context.Background(), DenoiseRequest{
			Steps: 50, Shift: 5, GuideScale: 6,
			CondContext: cond, UncondContext: uncond,
			Noise: sampling.CounterNoisePlan{
				Seed: 31, Grid: productionNoiseGrid(b, geometry.Elements()), Block: 256, Unroll: 4,
			},
		})
		if err != nil {
			_ = denoiser.Close()
			b.Fatal(err)
		}
		denoiseWall := time.Since(started)
		denoiseMemory, err := denoiser.MemoryStats()
		if err != nil {
			_ = denoiser.Close()
			b.Fatal(err)
		}
		if err := denoiser.ReleaseDenoiseResources(); err != nil {
			_ = denoiser.Close()
			b.Fatal(err)
		}
		testcheck.NoError(b, denoiser.Close())

		referencePath := filepath.Join(
			testutil.RepoRoot(b), "build", "latentvideo", "_r12_prod50", "final_latent.f32le",
		)
		if _, err := os.Stat(referencePath); err != nil {
			b.Fatalf("retained production latent unavailable: %v", err)
		}
		reference := loadF32LE(b, referencePath, geometry.Elements())
		cosine, normalizedRMS := latentAgreement(result.Latent, reference)
		// Retained run is an older BF16 approximation, not an F32 oracle. Keep
		// this cross-run bound tighter than the accepted BF16-vs-F32 G3 gate.
		if cosine < 0.999 || normalizedRMS > 0.03 {
			b.Fatalf("production latent agreement cosine=%.9f normalized_rms=%.6g", cosine, normalizedRMS)
		}

		totalWall := time.Since(started)
		peak := denoiseMemory.PeakBytes
		b.Logf(
			"Wan production denoise wall=%.2fs session+denoise=%.2fs peak=%d cosine=%.9f normalized_rms=%.6g",
			denoiseWall.Seconds(), totalWall.Seconds(), peak, cosine, normalizedRMS,
		)
		if totalWall.Seconds() >= wanOvergoDenoiseCeiling ||
			wanOvergoDenoiseCeiling+wanOvergoDecodeCeiling >= wanPythonProductionWall {
			b.Fatalf(
				"Wan denoise wall %.2fs or combined ceiling %.1fs fails Python %.1fs",
				totalWall.Seconds(), wanOvergoDenoiseCeiling+wanOvergoDecodeCeiling,
				wanPythonProductionWall,
			)
		}
		if peak > wanOvergoPeakCeiling {
			b.Fatalf("Wan production peak %d exceeds %d", peak, wanOvergoPeakCeiling)
		}
	}
}
