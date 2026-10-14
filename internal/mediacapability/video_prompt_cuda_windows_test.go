//go:build windows

package mediacapability

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"image/gif"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestVideoPromptActivation proves the page form end to end on the real
// Wan artifact: a request naming only a prompt, a seed and a small
// geometry runs through the video capability, the runtime resolves the
// conditioning contexts through the model's own text encoder and the noise
// plan from the seed, and the second run replays the first clip from the
// store. The request is asked for as a GIF, by naming no format, and as an
// MP4, which the same resident generator serves; each run's clip, the
// replayed one with it, publishes as the server publishes it.
func TestVideoPromptActivation(t *testing.T) {
	cudatest.Require(t)
	wan := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
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
	capability := videoCapability()
	execution := modelrecipetest.CandidateExecution(t, store, program)
	// The generator stays resident across formats, so its runs count on: the
	// GIF's first run is its first, and the MP4's first its second.
	for served, format := range []latentvideo.ClipFormat{"", "mp4"} {
		raw, err := json.Marshal(latentvideo.WanRequest{
			Prompt: "a red fox licking a vanilla ice cream cone on a snowy pine forest trail",
			Seed:   31, Frames: 5, Width: 32, Height: 16, Steps: 2, Format: format,
		})
		if err != nil {
			t.Fatal(err)
		}
		mediaType := map[latentvideo.ClipFormat]string{"": media.GIFMediaType, "mp4": media.MP4MediaType}[format]
		var first artifact.Content
		for run := 1; run <= 2; run++ {
			output, err := capability.Execute(t.Context(), store, wan, execution, string(raw))
			if err != nil {
				t.Fatal(err)
			}
			video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo)
			if !ok {
				t.Fatalf("video output type=%T", output)
			}
			frames := 0
			if format == "" {
				animation, err := gif.DecodeAll(bytes.NewReader(video.Data))
				if err != nil {
					t.Fatal(err)
				}
				frames = len(animation.Image)
			} else {
				shape, err := media.DecodeMP4Shape(video.Data)
				if err != nil {
					t.Fatal(err)
				}
				frames = shape.Frames
			}
			wantRun := 0
			if run == 1 {
				wantRun = served + 1
			}
			if video.MediaType != mediaType || video.Frames != 5 || frames != 5 || video.FPS != profile.SampleFPS ||
				video.Height != 16 || video.Width != 32 || video.ResidentRun != wantRun {
				t.Fatalf("format %q run %d video=%+v decoded_frames=%d", format, run, video, frames)
			}
			content, err := OutputContent(video)
			if err != nil {
				t.Fatalf("format %q run %d does not publish: %v", format, run, err)
			}
			if run == 1 {
				first = content
			} else if content.Descriptor != first.Descriptor {
				t.Fatalf("format %q replay published %s, the first run %s", format, content.Descriptor.ID, first.Descriptor.ID)
			}
			hash := sha256.Sum256(video.Data)
			t.Logf("Wan prompt activation format=%q run=%d %s=%dB hash=%s changed=%d", format, run, video.MediaType, len(video.Data), hex.EncodeToString(hash[:]), video.ChangedPixels)
		}
	}
}
