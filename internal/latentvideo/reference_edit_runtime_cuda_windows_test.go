//go:build windows

package latentvideo

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/testutil"
)

const (
	liveEditDeviceLatentOracle = "f73e55a69a2e5662100dfe664cc8234dc050d4b217174ce1a4ff45ac51bd4f17"
	liveEditDeviceFrameOracle  = "3dc33bc823a9a8083b0325cf78d78369e1f559943aa843b11afdaa477b902068"
)

func TestLiveEditDeviceRuntime(t *testing.T) {
	cudatest.Require(t)
	repo := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		t.Fatal(err)
	}
	wanDir := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	if _, err := os.Stat(filepath.Join(wanDir, "config.json")); os.IsNotExist(err) {
		t.Skipf("UNAVAILABLE: Wan denoiser config: %v", err)
	}
	policy := DenoiserPolicy{
		NumTrainTimesteps: 1000, SinusoidalPeriod: 10000, RotaryFrequencyBase: 10000,
		VAEStride: [3]int{4, 8, 8},
	}
	base, err := LoadDenoiserConfig(wanDir, policy)
	if err != nil {
		t.Fatal(err)
	}
	sigmas, err := CompileEditFlowSigmas(EditFlowConfig{
		InferenceSteps: 2, TrainTimesteps: policy.NumTrainTimesteps, Shift: 5,
		SigmaMin: 0, SigmaMax: 1, ExtraStep: true,
	}, []int64{900})
	if err != nil {
		t.Fatal(err)
	}
	context := readRetainedDenoiserF32(t, testutil.FixturePath(t, "wan", "raw", "g1_cond_context.f32le"), base.TextLen*base.Dim)
	runtime, err := NewReferenceEditRuntime(ReferenceEditRuntimeConfig{
		WanDirectory: wanDir, EditCheckpoint: filepath.Join(filepath.Dir(wanDir), "LiveEdit", "ar-forcing_002000.pt"),
		Policy: policy, LatentStats: loadSourceCodecStats(t),
		Source:         SourceVideoShape{Channels: 3, Frames: 5, Height: 16, Width: 32},
		FramesPerChunk: 1, LocalAttentionFrames: 1,
		Timesteps: []int64{900}, Sigmas: sigmas, ContextTimestep: 0,
		Layers: base.NumLayers, DeviceOrdinal: 0, TextContext: context,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	source := loadRealSourceCrops(t, filepath.Join(repo, "build", "latentvideo", "current_prod50", "frame_00.f32le"), runtime.sourcePlan.Source)
	initial := loadRetainedDenoiserCurrent(t)
	var priorFrameHash string
	for run := range 2 {
		var frames [][]float32
		started := time.Now()
		result, err := runtime.Run(t.Context(), source, initial, func(_ int, frame []float32, height, width int) error {
			if height != 16 || width != 32 {
				t.Fatalf("decoded frame shape=%dx%d", height, width)
			}
			frames = append(frames, append([]float32(nil), frame...))
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Sampler.DenoiseCalls != 2 || result.Sampler.ContextRefreshCalls != 2 ||
			result.Source.SourceChunks != 2 || result.Decode.OutputFrames != 5 || len(frames) != 5 {
			t.Fatalf("runtime result sampler=%+v source=%+v decode=%+v frames=%d", result.Sampler, result.Source, result.Decode, len(frames))
		}
		latentHash := hashReferenceEditValues(result.Latent)
		frameHash := hashReferenceEditFrames(frames)
		if latentHash != liveEditDeviceLatentOracle || frameHash != liveEditDeviceFrameOracle {
			t.Fatalf("device runtime oracle latent=%s frames=%s", latentHash, frameHash)
		}
		if result.Denoiser.Layers != base.NumLayers || result.Denoiser.ContextProjections != 1 || result.Denoiser.Runs != (run+1)*4 {
			t.Fatalf("device denoiser lifecycle=%+v", result.Denoiser)
		}
		if run > 0 && frameHash != priorFrameHash {
			t.Fatalf("resident replay frame hash=%s want=%s", frameHash, priorFrameHash)
		}
		priorFrameHash = frameHash
		for _, values := range append([][]float32{result.Latent}, frames...) {
			for _, value := range values {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatal("reference edit runtime emitted non-finite values")
				}
			}
		}
		t.Logf("LiveEdit device runtime run=%d wall=%.3fs latent=%s frames=%s encode=%.3fs decode=%.3fs denoiser=%+v peaks=%.3f/%.3f/%.3fGiB",
			run, time.Since(started).Seconds(), latentHash, frameHash, result.Source.WallSeconds, result.Decode.DecodeWallSec, result.Denoiser,
			float64(result.Source.PeakDeviceBytes)/(1<<30), float64(result.DenoiseMemory.PeakBytes)/(1<<30), float64(result.DecoderMemory.PeakBytes)/(1<<30))
	}
}

func hashReferenceEditValues(values []float32) string {
	hash := sha256.New()
	var raw [4]byte
	for _, value := range values {
		binary.LittleEndian.PutUint32(raw[:], math.Float32bits(value))
		_, _ = hash.Write(raw[:])
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func hashReferenceEditFrames(frames [][]float32) string {
	hash := sha256.New()
	for _, frame := range frames {
		value, _ := hex.DecodeString(hashReferenceEditValues(frame))
		_, _ = hash.Write(value)
	}
	return hex.EncodeToString(hash.Sum(nil))
}
