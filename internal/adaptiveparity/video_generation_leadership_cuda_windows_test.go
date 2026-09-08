//go:build windows

package adaptiveparity

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/latentvideo"
	"overgo/internal/testutil"
)

const (
	pythonLiveEditWallSeconds   = 60.1
	adaptiveLiveEditWallSeconds = 109.5
	adaptiveLiveEditPeakBytes   = uint64(20_609_153_488)
	liveEditLatentCosineFloor   = 0.998
	liveEditLatentNRMSCeiling   = 0.06
)

func TestVideoGenerationLeadership(t *testing.T) {
	cudatest.Require(t)
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	repo := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		t.Fatal(err)
	}
	wan := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	evidence := filepath.Join(filepath.Dir(roots.Models), ".claude", "worktrees", "image_gen", ".media_artifacts", "video_artifacts")
	ffmpeg := filepath.Join(filepath.Dir(roots.Models), ".env", "Lib", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg-win-x86_64-v7.1.exe")
	probes := filepath.Join(evidence, "liveedit_native_go_currants_seed42.mp4.tensors")
	requireLeadershipSHA256(t, filepath.Join(probes, "text_context.f32le"), "13d75474179a0041325973fd8f52df4110fb8d26c8b9ba71d4445dfc39cb41be")
	requireLeadershipSHA256(t, filepath.Join(probes, "source_pixels.f32le"), "e0c1c32d73050cf68cdd568679bb7e2cb127ab158cf6b98d8169b65d140e1087")
	requireLeadershipSHA256(t, filepath.Join(probes, "final_latent.f32le"), "46a1844932ac83b355c143fc21a9e4d70ec12e0074c1c7c228e10425f87f1049")
	requireLeadershipSHA256(t, filepath.Join(evidence, "liveedit_native_no_pruning_current_seed42.mp4"), "a49e89e3999e95c7ffc02964188b22c7e712647f51f6aa4402c219de82fba53b")
	requireLeadershipSHA256(t, filepath.Join(evidence, "liveedit_python_oracle_current", "liveedit_native_python_seed42.mp4"), "ecca8c316cc0ce7e090f2a918b326bd74e4eeaaa95d5c923b8bfd13ebbba7950")
	context := readLeadershipF32(t, filepath.Join(probes, "text_context.f32le"), 512*1536)
	source := readLeadershipF32(t, filepath.Join(probes, "source_pixels.f32le"), 3*81*480*832)
	wantLatent := readLeadershipF32(t, filepath.Join(probes, "final_latent.f32le"), 16*21*60*104)
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	sigmas, err := latentvideo.CompileEditFlowSigmas(latentvideo.EditFlowConfig{
		InferenceSteps: 1000, TrainTimesteps: profile.Policy.NumTrainTimesteps,
		Shift: 5, SigmaMin: 0, SigmaMax: 1, ExtraStep: true,
	}, []int64{1000, 750, 500, 250})
	if err != nil {
		t.Fatal(err)
	}
	loadStarted := time.Now()
	runtime, err := latentvideo.NewReferenceEditRuntime(latentvideo.ReferenceEditRuntimeConfig{
		WanDirectory: wan, EditCheckpoint: filepath.Join(roots.Models, "LiveEdit", "ar-forcing_002000.pt"),
		Policy: profile.Policy, LatentStats: profile.LatentStats,
		Source:         latentvideo.SourceVideoShape{Channels: 3, Frames: 81, Height: 480, Width: 832},
		FramesPerChunk: 3, LocalAttentionFrames: 21,
		Timesteps: []int64{1000, 750, 500, 250}, Sigmas: sigmas,
		Layers: 30, DeviceOrdinal: 0, Seed: 42, TextContext: context,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	loadWall := time.Since(loadStarted)
	var firstVideo latentvideo.EncodedVideo
	var firstLatent []float32
	var coldWall, warmWall time.Duration
	var peak uint64
	for run := range 2 {
		encoder, err := latentvideo.NewMP4Encoder(t.Context(), ffmpeg, profile.SampleFPS, 480, 832, latentvideo.UnitPixels)
		if err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		result, err := runtime.Run(t.Context(), source, nil, encoder.Add)
		wall := time.Since(started)
		if err != nil {
			_ = encoder.Close()
			t.Fatal(err)
		}
		video, err := encoder.Finish()
		if err != nil {
			_ = encoder.Close()
			t.Fatal(err)
		}
		if run == 0 {
			firstVideo, coldWall = video, wall
		} else {
			warmWall = wall
			replayCosine, replayNRMS := leadershipAgreement(result.Latent, firstLatent)
			if replayCosine < 0.9999 || replayNRMS > 0.02 {
				t.Fatalf("LiveEdit resident replay cosine=%.9f nrms=%.6f", replayCosine, replayNRMS)
			}
		}
		cosine, nrms := leadershipAgreement(result.Latent, wantLatent)
		peak = max(peak, result.Source.PeakDeviceBytes, result.DenoiseMemory.PeakBytes, result.DecoderMemory.PeakBytes)
		t.Logf("LiveEdit full run=%d wall=%.3fs encode=%.3fs cache=%t denoise=%.3fs decode=%.3fs latent_cosine=%.9f nrms=%.6f frames=%d mp4=%.3fMiB peak=%.3fGiB",
			run, wall.Seconds(), result.Source.WallSeconds, result.Source.CacheHit, result.DenoiseWallSec,
			result.Decode.DecodeWallSec, cosine, nrms, video.Frames, float64(len(video.Data))/(1<<20), float64(peak)/(1<<30))
		if cosine < liveEditLatentCosineFloor || nrms > liveEditLatentNRMSCeiling {
			t.Fatalf("LiveEdit final latent cosine=%.9f nrms=%.6f", cosine, nrms)
		}
		if run == 0 {
			if result.Source.CacheHit {
				t.Fatal("LiveEdit cold source unexpectedly hit cache")
			}
			firstLatent = append([]float32(nil), result.Latent...)
		} else if !result.Source.CacheHit || result.Source.WallSeconds != 0 {
			t.Fatalf("LiveEdit warm source cache hit=%t wall=%.3fs", result.Source.CacheHit, result.Source.WallSeconds)
		}
		t.Logf("LiveEdit MP4 sha256=%x", sha256.Sum256(video.Data))
	}
	if firstVideo.Frames != 81 || firstVideo.Height != 480 || firstVideo.Width != 832 {
		t.Fatalf("LiveEdit clip=%+v", firstVideo)
	}
	for _, reference := range []struct {
		name string
		path string
	}{
		{"adaptive", filepath.Join(evidence, "liveedit_native_no_pruning_current_seed42.mp4")},
		{"python", filepath.Join(evidence, "liveedit_python_oracle_current", "liveedit_native_python_seed42.mp4")},
	} {
		metrics := compareLeadershipVideo(t, ffmpeg, firstVideo.Data, reference.path, firstVideo.Frames, firstVideo.Height, firstVideo.Width)
		t.Logf("LiveEdit conditioned %s agreement mae=%.6f psnr=%.3fdB temporal_mad=%.6f reference_temporal_mad=%.6f ratio=%.3f",
			reference.name, metrics.MAE, metrics.PSNR, metrics.TemporalMAD, metrics.ReferenceTemporalMAD, metrics.TemporalRatio)
		if metrics.MAE > 0.02 || metrics.PSNR < 28 || metrics.TemporalRatio < 0.85 || metrics.TemporalRatio > 1.15 {
			t.Fatalf("LiveEdit conditioned %s quality=%+v", reference.name, metrics)
		}
	}
	if coldWall.Seconds() >= adaptiveLiveEditWallSeconds {
		t.Fatalf("LiveEdit cold wall=%.3fs does not beat adaptive %.1fs", coldWall.Seconds(), adaptiveLiveEditWallSeconds)
	}
	if warmWall.Seconds() >= pythonLiveEditWallSeconds {
		t.Fatalf("LiveEdit warm wall=%.3fs does not beat Python %.1fs", warmWall.Seconds(), pythonLiveEditWallSeconds)
	}
	if peak > adaptiveLiveEditPeakBytes {
		t.Fatalf("LiveEdit peak=%d exceeds adaptive=%d", peak, adaptiveLiveEditPeakBytes)
	}
	t.Logf("LiveEdit lifecycle load=%.3fs cold=%.3fs warm=%.3fs peak=%.3fGiB", loadWall.Seconds(), coldWall.Seconds(), warmWall.Seconds(), float64(peak)/(1<<30))
}

type leadershipVideoMetrics struct {
	MAE, PSNR, TemporalMAD, ReferenceTemporalMAD, TemporalRatio float64
}

func compareLeadershipVideo(t testing.TB, ffmpeg string, candidate []byte, reference string, frames, height, width int) leadershipVideoMetrics {
	t.Helper()
	candidateCommand, candidateOutput, candidateErrors := startLeadershipDecoder(t, ffmpeg, "pipe:0", bytes.NewReader(candidate))
	referenceCommand, referenceOutput, referenceErrors := startLeadershipDecoder(t, ffmpeg, reference, nil)
	defer candidateOutput.Close()
	defer referenceOutput.Close()
	frameBytes := 3 * height * width
	candidateFrame, referenceFrame := make([]byte, frameBytes), make([]byte, frameBytes)
	previousCandidate, previousReference := make([]byte, frameBytes), make([]byte, frameBytes)
	var absolute, squared, temporal, referenceTemporal float64
	for frame := range frames {
		if _, err := io.ReadFull(candidateOutput, candidateFrame); err != nil {
			t.Fatalf("candidate frame %d: %v: %s", frame, err, candidateErrors.String())
		}
		if _, err := io.ReadFull(referenceOutput, referenceFrame); err != nil {
			t.Fatalf("reference frame %d: %v: %s", frame, err, referenceErrors.String())
		}
		for index, expected := range referenceFrame {
			delta := math.Abs(float64(candidateFrame[index]) - float64(expected))
			absolute += delta
			squared += delta * delta
			if frame > 0 {
				temporal += math.Abs(float64(candidateFrame[index]) - float64(previousCandidate[index]))
				referenceTemporal += math.Abs(float64(expected) - float64(previousReference[index]))
			}
		}
		copy(previousCandidate, candidateFrame)
		copy(previousReference, referenceFrame)
	}
	for name, item := range map[string]struct {
		command *exec.Cmd
		output  io.Reader
		errors  *boundedLeadershipError
	}{"candidate": {candidateCommand, candidateOutput, candidateErrors}, "reference": {referenceCommand, referenceOutput, referenceErrors}} {
		var extra [1]byte
		if count, err := item.output.Read(extra[:]); count != 0 || !errors.Is(err, io.EOF) {
			t.Fatalf("%s video has trailing frame data count=%d err=%v", name, count, err)
		}
		if err := item.command.Wait(); err != nil {
			t.Fatalf("%s decoder: %v: %s", name, err, item.errors.String())
		}
	}
	comparisons := float64(frames * frameBytes)
	pairs := float64((frames - 1) * frameBytes)
	meanSquared := squared / comparisons
	metrics := leadershipVideoMetrics{
		MAE:                  absolute / comparisons / 255,
		PSNR:                 20 * math.Log10(255/math.Sqrt(meanSquared)),
		TemporalMAD:          temporal / pairs / 255,
		ReferenceTemporalMAD: referenceTemporal / pairs / 255,
	}
	metrics.TemporalRatio = metrics.TemporalMAD / metrics.ReferenceTemporalMAD
	return metrics
}

func startLeadershipDecoder(t testing.TB, ffmpeg, input string, data io.Reader) (*exec.Cmd, io.ReadCloser, *boundedLeadershipError) {
	t.Helper()
	command := exec.Command(ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", input, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")
	command.Stdin = data
	errors := &boundedLeadershipError{}
	command.Stderr = errors
	output, err := command.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	return command, output, errors
}

type boundedLeadershipError struct{ data []byte }

func (w *boundedLeadershipError) Write(data []byte) (int, error) {
	written := len(data)
	if remaining := 64<<10 - len(w.data); remaining > 0 {
		w.data = append(w.data, data[:min(len(data), remaining)]...)
	}
	return written, nil
}

func (w *boundedLeadershipError) String() string { return string(w.data) }

func requireLeadershipSHA256(t testing.TB, path, want string) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("UNAVAILABLE: video leadership evidence %s: %v", path, err)
	}
	digest := sha256.New()
	_, copyErr := io.Copy(digest, file)
	closeErr := file.Close()
	if copyErr != nil || closeErr != nil {
		t.Fatalf("video leadership evidence hash %s: copy=%v close=%v", path, copyErr, closeErr)
	}
	if got := fmt.Sprintf("%x", digest.Sum(nil)); got != want {
		t.Fatalf("video leadership evidence %s sha256=%s want=%s", path, got, want)
	}
}

func readLeadershipF32(t testing.TB, path string, elements int) []float32 {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("UNAVAILABLE: video leadership evidence %s: %v", path, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.Size() != int64(elements)*4 {
		t.Fatalf("video leadership evidence size=%d want=%d err=%v", info.Size(), int64(elements)*4, err)
	}
	values := make([]float32, elements)
	buffer := make([]byte, 4<<20)
	for offset := 0; offset < elements; {
		count := min(len(buffer)/4, elements-offset)
		raw := buffer[:count*4]
		if _, err := io.ReadFull(file, raw); err != nil {
			t.Fatal(err)
		}
		for index := range count {
			values[offset+index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
		}
		offset += count
	}
	return values
}

func leadershipAgreement(got, want []float32) (cosine, normalizedRMS float64) {
	if len(got) != len(want) || len(got) == 0 {
		return 0, math.Inf(1)
	}
	var dot, gotNorm, wantNorm, deltaNorm float64
	for index, expected := range want {
		actual := float64(got[index])
		reference := float64(expected)
		dot += actual * reference
		gotNorm += actual * actual
		wantNorm += reference * reference
		delta := actual - reference
		deltaNorm += delta * delta
	}
	return dot / math.Sqrt(gotNorm*wantNorm), math.Sqrt(deltaNorm / wantNorm)
}
