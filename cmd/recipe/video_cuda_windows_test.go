//go:build windows

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"image/gif"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestVideoProductionActivation(t *testing.T) {
	cudatest.Require(t)
	repo := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		t.Fatal(err)
	}
	wan := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	edit := filepath.Join(roots.Models, "LiveEdit")
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "liveedit-production-activation")
	definition, err := modelrecipe.ReferenceVideoEditDefinition(modelID, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	testutil.PublishArtifact(t, store, modelID)
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "test/video-profile", Contents: []artifact.Content{profileContent},
	}); err != nil {
		t.Fatal(err)
	}

	base, err := latentvideo.LoadDenoiserConfig(wan, profile.Policy)
	if err != nil {
		t.Fatal(err)
	}
	sigmas, err := latentvideo.CompileEditFlowSigmas(latentvideo.EditFlowConfig{
		InferenceSteps: 2, TrainTimesteps: profile.Policy.NumTrainTimesteps,
		Shift: 5, SigmaMin: 0, SigmaMax: 1, ExtraStep: true,
	}, []int64{900})
	if err != nil {
		t.Fatal(err)
	}
	request := latentvideo.ReferenceEditRequest{
		Condition: latentvideo.ReferenceEditCondition{
			TextContext:  readVideoF32(t, testutil.FixturePath(t, "wan", "raw", "g1_cond_context.f32le"), base.TextLen*base.Dim),
			InitialNoise: readVideoInitialNoise(t), FramesPerChunk: 1, LocalAttention: 1,
			Timesteps: []int64{900}, Sigmas: sigmas, ContextTimestep: 0,
		},
		Source: latentvideo.SourceVideo{
			Pixels:   loadVideoSourceCrops(t, filepath.Join(repo, "build", "latentvideo", "current_prod50", "frame_00.f32le"), 5, 16, 32),
			Channels: 3, Frames: 5, Height: 16, Width: 32,
		},
	}
	raw, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	capability := videoCapability()
	var firstHash string
	for run := 1; run <= 2; run++ {
		output, err := capability.execute(t.Context(), store, edit, modelID, program, string(raw))
		if err != nil {
			t.Fatal(err)
		}
		video, ok := output.(latentvideo.EncodedVideo)
		if !ok {
			t.Fatalf("video output type=%T", output)
		}
		animation, err := gif.DecodeAll(bytes.NewReader(video.Data))
		if err != nil {
			t.Fatal(err)
		}
		if video.MediaType != "image/gif" || video.Frames != 5 || len(animation.Image) != 5 ||
			video.Height != 16 || video.Width != 32 || video.ResidentRun != run {
			t.Fatalf("run %d video=%+v decoded_frames=%d", run, video, len(animation.Image))
		}
		hash := sha256.Sum256(video.Data)
		encodedHash := hex.EncodeToString(hash[:])
		if run == 1 {
			firstHash = encodedHash
		} else if encodedHash != firstHash {
			t.Fatalf("resident replay hash=%s want=%s", encodedHash, firstHash)
		}
		t.Logf("LiveEdit production activation run=%d gif=%dB hash=%s changed=%d", run, len(video.Data), encodedHash, video.ChangedPixels)
	}
}

func readVideoF32(t testing.TB, path string, elements int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != elements*4 {
		t.Fatalf("float fixture bytes=%d want=%d", len(raw), elements*4)
	}
	values := make([]float32, elements)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*index:]))
	}
	return values
}

func readVideoInitialNoise(t testing.TB) []float32 {
	t.Helper()
	raw, err := os.ReadFile(testutil.FixturePath(t, "wan", "g4_denoise.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Final struct {
			Values []float32 `json:"values"`
		} `json:"final_latent"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Final.Values) != 16*2*2*4 {
		t.Fatalf("initial noise elements=%d", len(fixture.Final.Values))
	}
	return fixture.Final.Values
}

func loadVideoSourceCrops(t testing.TB, path string, frames, height, width int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("UNAVAILABLE: retained real Wan frame: %v", err)
	}
	const sourceHeight, sourceWidth = 480, 832
	if len(raw) != 3*sourceHeight*sourceWidth*4 {
		t.Fatalf("real source bytes=%d", len(raw))
	}
	output := make([]float32, 3*frames*height*width)
	for channel := range 3 {
		for frame := range frames {
			originY, originX := 32*frame, 48*frame
			for y := range height {
				for x := range width {
					source := channel*sourceHeight*sourceWidth + (originY+y)*sourceWidth + originX + x
					target := ((channel*frames+frame)*height+y)*width + x
					output[target] = math.Float32frombits(binary.LittleEndian.Uint32(raw[source*4:]))
				}
			}
		}
	}
	return output
}
