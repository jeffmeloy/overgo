//go:build windows

package latentvideo

import (
	"cmp"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/pytorchzip"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// Same f64 host versus fp32 device accumulation envelope as decoder captures.
const sourceCodecTolerance = 1e-5

func TestLiveEditSourceCodec(t *testing.T) {
	cudatest.Require(t)
	repo := testutil.RepoRoot(t)
	roots := testcheck.Must(dataroot.Resolve(repo))(t)
	checkpoint := filepath.Join(roots.ResolveModelPath("Wan2.1-T2V-1.3B"), "Wan2.1_VAE.pth")
	if _, err := os.Stat(checkpoint); os.IsNotExist(err) {
		t.Skipf("UNAVAILABLE: Wan source codec: %v", err)
	}
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		t.Fatalf("UNAVAILABLE: Wan source codec: %v", err)
	}
	graph := testcheck.Must(CompileVAEEncoderPlan(catalog.Tensors))(t)
	stats := loadSourceCodecStats(t)
	profile := SourceCodecProfile{InputChannels: graph.InputChannels, LatentChannels: graph.LatentChannels, Stride: graph.Stride, LatentStats: stats}
	plan := testcheck.Must(CompileSourceCodecBoundary(profile, SourceVideoShape{Channels: 3, Frames: 5, Height: 32, Width: 32}))(t)
	source := loadRealSourceCrops(t, plan.Source)
	hostStarted := time.Now()
	want := testcheck.Must(EncodeSourceVideo(checkpoint, graph, plan, source))(t)
	hostWall := time.Since(hostStarted)
	session := testcheck.Must(newVAEEncoderCUDASession(checkpoint, graph, 0))(t)
	defer session.Close()
	got, evidence, err := session.Encode(t.Context(), plan, source)
	testcheck.NoError(t, err)
	maximum := sourceCodecMaxAbs(got, want)
	var energy float64
	for _, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatal("source latent is non-finite")
		}
		energy += math.Abs(float64(value))
	}
	t.Logf("LiveEdit source codec: source=%dx%dx%d latent=%dx%dx%d chunks=%d host=%s cuda=%.3fs weights=%.3fMiB peak=%.3fMiB max=%.3e",
		plan.Source.Frames, plan.Source.Height, plan.Source.Width, plan.Latent.Frames, plan.Latent.Height, plan.Latent.Width,
		evidence.SourceChunks, hostWall, evidence.WallSeconds, float64(evidence.WeightBytes)/(1<<20), float64(evidence.PeakDeviceBytes)/(1<<20), maximum)
	if energy == 0 || maximum > sourceCodecTolerance {
		t.Fatalf("source codec parity max=%.3e energy=%.3e", maximum, energy)
	}
}

func loadSourceCodecStats(t testing.TB) VAELatentStats {
	t.Helper()
	raw := testcheck.Must(os.ReadFile(testutil.FixturePath(t, "wan", "g0_config.json")))(t)
	var fixture struct {
		Stats struct {
			Mean []float32 `json:"mean"`
			Std  []float32 `json:"std"`
		} `json:"vae_latent_stats"`
	}
	testcheck.NoError(t, json.Unmarshal(raw, &fixture))
	return VAELatentStats{Mean: fixture.Stats.Mean, Std: fixture.Stats.Std}
}

func loadRealSourceCrops(t testing.TB, shape SourceVideoShape) []float32 {
	t.Helper()
	// The gate binds the original data root while source runs in a candidate.
	// This retained capture is data, not an output to regenerate in that tree.
	base := cmp.Or(os.Getenv(dataroot.Env), testutil.RepoRoot(t))
	path := filepath.Join(base, "build", "latentvideo", "current_prod50", "frame_00.f32le")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("required Wan source capture at data root %s: %v", base, err)
	}
	// Identity of the original 832x480 production capture used by native oracles.
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != "ce3637bfb13020731b5158672e6100c9f2d936cc3588394a0f00040e50b516f6" {
		t.Fatal("retained Wan source capture changed")
	}
	const frameHeight, frameWidth = 480, 832
	if len(raw) != 3*frameHeight*frameWidth*4 {
		t.Fatalf("real source bytes=%d", len(raw))
	}
	output := make([]float32, shape.Channels*shape.Frames*shape.Height*shape.Width)
	for channel := range shape.Channels {
		for frame := range shape.Frames {
			originY, originX := 32*frame, 48*frame
			for y := range shape.Height {
				for x := range shape.Width {
					sourceIndex := channel*frameHeight*frameWidth + (originY+y)*frameWidth + originX + x
					destination := ((channel*shape.Frames+frame)*shape.Height+y)*shape.Width + x
					output[destination] = math.Float32frombits(binary.LittleEndian.Uint32(raw[sourceIndex*4:]))
				}
			}
		}
	}
	return output
}

func sourceCodecMaxAbs(left, right []float32) float64 {
	if len(left) != len(right) {
		return math.Inf(1)
	}
	maximum := float64(0)
	for index := range left {
		maximum = max(maximum, math.Abs(float64(left[index])-float64(right[index])))
	}
	return maximum
}
