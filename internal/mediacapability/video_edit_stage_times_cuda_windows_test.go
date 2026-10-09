//go:build windows

package mediacapability

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/testutil"
)

// unnamedShare is the part of a served edit's time that may lie outside every
// named part of it.
const unnamedShare = 0.05

// servedEditCeiling is the time a served edit of 21 frames may take. It took
// 75 seconds while the text encoder ran a prompt's padding with the prompt,
// and takes 41 without; the ceiling stands between the two, so that cost
// coming back fails here. The model's own code takes four seconds over the
// same clip on this device, and each part mended brings the ceiling down.
const servedEditCeiling = 64 * time.Second

// TestServedEditStageTimes says where a served edit's time goes. It serves the
// edit of one clip of 21 frames at the size the editor was trained at, the
// clip the held-out measure cuts, three times: with nothing loaded, then with
// another instruction, then with another seed. Each request's time is divided
// among the named parts of the path it takes, and the test fails while the
// parts leave more than a twentieth of a request unnamed: a cost no part names
// is one nobody is looking at. The parts the model's own code has took, on the
// same device over a clip of the same size (scripts/python_golden_references/
// liveedit_own_edit.py): its text encoder 0.1 to 0.2 seconds, its source
// encoding 0.7 to 1.0, its denoising 2.1 to 2.2 and its decoding 1.25.
func TestServedEditStageTimes(t *testing.T) {
	cudatest.Require(t)
	edit := modeltest.Directory(t, "LiveEdit")
	wan, checkpoint := latentvideo.LiveEditArtifacts(edit)
	for _, required := range []string{checkpoint, latentvideo.WanTextConditioningSpec(wan, 1).EncoderCheckpoint} {
		if _, err := os.Stat(required); err != nil {
			t.Skipf("UNAVAILABLE: %v", err)
		}
	}
	width, height, err := latentvideo.EditSourceSize(edit)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "liveedit-stage-times")
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
	const frames = 21
	plane := height * width
	source := latentvideo.SourceVideo{
		Pixels: make([]float32, media.RGBChannels*frames*plane), Channels: media.RGBChannels, Frames: frames, Height: height, Width: width,
	}
	for channel := range media.RGBChannels {
		for frame := range frames {
			for at := range plane {
				x, y := at%width, at/width
				source.Pixels[(channel*frames+frame)*plane+at] = float32((x+2*y+7*frame+40*channel)%width) / float32(width)
			}
		}
	}
	clip, err := latentvideo.SourceFramesContent(source)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/video-edit-stage-times", Contents: []artifact.Content{profileContent, clip},
	}); err != nil {
		t.Fatal(err)
	}
	capability := videoCapability()
	execution := modelrecipetest.CandidateExecution(t, store, program)
	for _, request := range []struct {
		name string
		form latentvideo.ReferenceEditRequest
	}{
		{"nothing loaded", latentvideo.ReferenceEditRequest{Prompt: "turn the stripes green", Seed: 3, SourceArtifact: clip.Descriptor.ID}},
		{"another instruction", latentvideo.ReferenceEditRequest{Prompt: "turn the stripes into falling snow", Seed: 3, SourceArtifact: clip.Descriptor.ID}},
		{"another seed", latentvideo.ReferenceEditRequest{Prompt: "turn the stripes into falling snow", Seed: 4, SourceArtifact: clip.Descriptor.ID}},
	} {
		raw, err := json.Marshal(request.form)
		if err != nil {
			t.Fatal(err)
		}
		ctx, stages := processmeasure.WithStages(t.Context())
		clock := processmeasure.NewStopwatch()
		output, err := capability.Execute(ctx, store, edit, execution, string(raw))
		elapsed, clockErr := clock.Elapsed()
		if err != nil || clockErr != nil {
			t.Fatal(err, clockErr)
		}
		if video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo); !ok || video.Frames != frames || video.Width != width || video.Height != height {
			t.Fatalf("%s: a served edit is %T %+v", request.name, output, video)
		}
		wall := time.Duration(elapsed)
		var report strings.Builder
		unnamed, err := stages.Report(&report, wall)
		if err != nil {
			t.Fatalf("%s: %v", request.name, err)
		}
		t.Logf("%s: %.2f s\n%s", request.name, wall.Seconds(), report.String())
		if unnamed.Seconds() > unnamedShare*wall.Seconds() || unnamed < 0 {
			t.Errorf("%s: no part names %.2f s of a request of %.2f s", request.name, unnamed.Seconds(), wall.Seconds())
		}
		if wall > servedEditCeiling {
			t.Errorf("%s: the edit took %.2f s, over the %v it is held to", request.name, wall.Seconds(), servedEditCeiling)
		}
	}
}
