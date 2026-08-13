//go:build windows

package latentvideo

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor/dtype"
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
	library, err := driver.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		t.Fatal(err)
	}
	info, err := library.DeviceInfo(0)
	if err != nil {
		t.Fatal(err)
	}
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
	geometry, err := config.CompileLatentGeometry(81, 832, 480)
	if err != nil {
		b.Fatal(err)
	}
	program, err := CompileDenoiserProgramPrecision(config, weights, geometry, DenoiserPrecision{
		MatmulWeights: dtype.BF16, RoundAttentionStorage: true,
	})
	if err != nil {
		b.Fatal(err)
	}
	fixtureDir := testutil.FixturePath(b, "wan")
	contextElements := config.TextLen * config.Dim
	cond := loadRawContext(b, fixtureDir, "raw/g1_cond_context.f32le", contextElements)
	uncond := loadRawContext(b, fixtureDir, "raw/g1_uncond_context.f32le", contextElements)

	b.ResetTimer()
	for range b.N {
		started := time.Now()
		denoiser, err := NewDenoiserCUDASession(program, 0)
		if err != nil {
			b.Fatal(err)
		}
		result, err := denoiser.Denoise(context.Background(), DenoiseRequest{
			Steps: 50, Shift: 5, GuideScale: 6,
			CondContext: cond, UncondContext: uncond,
			Noise: NoisePlan{
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
		if err := denoiser.Close(); err != nil {
			b.Fatal(err)
		}

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

func latentAgreement(got, want []float32) (cosine, normalizedRMS float64) {
	var dot, gotNorm, wantNorm, deltaNorm float64
	for index := range want {
		g, w := float64(got[index]), float64(want[index])
		dot += g * w
		gotNorm += g * g
		wantNorm += w * w
		delta := g - w
		deltaNorm += delta * delta
	}
	return dot / math.Sqrt(gotNorm*wantNorm), math.Sqrt(deltaNorm / wantNorm)
}
