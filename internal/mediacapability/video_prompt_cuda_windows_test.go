//go:build windows

package mediacapability

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"image/gif"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestVideoPromptActivation proves the page form end to end on the real
// Wan artifact: a request naming only a prompt, a seed and a small
// geometry runs through the video capability, the runtime resolves the
// conditioning contexts through the model's own text encoder and the noise
// plan from the seed, and the resident second run replays the first clip.
func TestVideoPromptActivation(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		t.Fatal(err)
	}
	wan := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	spec := latentvideo.WanTextConditioningSpec(wan, 1)
	if _, err := os.Stat(spec.EncoderCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: encoder checkpoint absent: %v", err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "wan-prompt-activation")
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, modelID, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishArtifactExtent(t, store, modelID, wan)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/video-prompt-profile", Contents: []artifact.Content{profileContent},
	}); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(latentvideo.WanRequest{
		Prompt: "a red fox licking a vanilla ice cream cone on a snowy pine forest trail",
		Seed:   31, Frames: 5, Width: 32, Height: 16, Steps: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	capability := videoCapability()
	execution := candidateCapabilityExecution(t, store, program)
	var firstHash string
	for run := 1; run <= 2; run++ {
		output, err := capability.Execute(t.Context(), store, wan, execution, string(raw))
		if err != nil {
			t.Fatal(err)
		}
		video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo)
		if !ok {
			t.Fatalf("video output type=%T", output)
		}
		animation, err := gif.DecodeAll(bytes.NewReader(video.Data))
		if err != nil {
			t.Fatal(err)
		}
		if video.MediaType != "image/gif" || video.Frames != 5 || len(animation.Image) != 5 ||
			video.Height != 16 || video.Width != 32 || video.ResidentRun != residentRun(run) {
			t.Fatalf("run %d video=%+v decoded_frames=%d", run, video, len(animation.Image))
		}
		hash := sha256.Sum256(video.Data)
		encodedHash := hex.EncodeToString(hash[:])
		if run == 1 {
			firstHash = encodedHash
		} else if encodedHash != firstHash {
			t.Fatalf("resident replay hash=%s want=%s", encodedHash, firstHash)
		}
		t.Logf("Wan prompt activation run=%d gif=%dB hash=%s changed=%d", run, len(video.Data), encodedHash, video.ChangedPixels)
	}
}
