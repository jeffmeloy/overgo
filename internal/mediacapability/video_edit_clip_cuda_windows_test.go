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
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestVideoEditClipActivation proves the clip form end to end on the real
// LiveEdit artifact: a GIF clip committed to the store, a request naming
// only a prompt, a seed and that clip's artifact id, the capability
// resolving the declared edit schedule, the text context through the Wan
// text pipeline and the pixels from the clip, and the resident second run
// replaying the first edit.
func TestVideoEditClipActivation(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	edit := filepath.Join(roots.Models, "LiveEdit")
	wan, checkpoint := latentvideo.LiveEditArtifacts(edit)
	for _, required := range []string{checkpoint, latentvideo.WanTextConditioningSpec(wan, 1).EncoderCheckpoint} {
		if _, err := os.Stat(required); err != nil {
			t.Skipf("UNAVAILABLE: %v", err)
		}
	}
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "liveedit-clip-activation")
	definition, err := modelrecipe.ReferenceVideoEditDefinition(modelID, profile.ID)
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
	publishArtifactExtent(t, store, modelID, edit)
	// 21 source frames are 6 latent frames: two chunks of the declared
	// three, inside the local attention history.
	clip := syntheticClipContent(t, 21, 16, 32)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/video-edit-clip", Contents: []artifact.Content{profileContent, clip},
	}); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(latentvideo.ReferenceEditRequest{
		Prompt: "a red fox licking a vanilla ice cream cone on a snowy pine forest trail",
		Seed:   7, SourceArtifact: clip.Descriptor.ID.String(),
	})
	if err != nil {
		t.Fatal(err)
	}
	capability := videoCapability()
	execution := modelrecipetest.CandidateExecution(t, store, program)
	var firstHash string
	for run := 1; run <= 2; run++ {
		output, err := capability.Execute(t.Context(), store, edit, execution, string(raw))
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
		if video.MediaType != media.GIFMediaType || video.Frames != 21 || len(animation.Image) != 21 ||
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
		t.Logf("LiveEdit clip activation run=%d gif=%dB hash=%s changed=%d", run, len(video.Data), encodedHash, video.ChangedPixels)
	}
}

// syntheticClipContent encodes a moving gradient as the production GIF the
// page would attach: frames x height x width, signed-unit planar pixels.
func syntheticClipContent(t testing.TB, frames, height, width int) artifact.Content {
	t.Helper()
	encoder, err := latentvideo.NewGIFEncoder(16, latentvideo.SignedUnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	plane := height * width
	for frame := range frames {
		values := make([]float32, media.RGBChannels*plane)
		for channel := range media.RGBChannels {
			for y := range height {
				for x := range width {
					values[channel*plane+y*width+x] = float32((x+y+frame*2+channel*5)%width)/float32(width)*2 - 1
				}
			}
		}
		if err := encoder.Add(frame, values, height, width); err != nil {
			t.Fatal(err)
		}
	}
	video, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	content, err := latentvideo.GIFContent(video)
	if err != nil {
		t.Fatal(err)
	}
	return content
}
