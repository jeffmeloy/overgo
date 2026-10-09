//go:build windows

package mediacapability

import (
	"strings"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentvideo"
	"overgo/internal/modeltest"
	"overgo/internal/processmeasure"
)

// unnamedShare is the part of a served edit's time that may lie outside every
// named part of it.
const unnamedShare = 0.05

// servedEditCeiling is the time a served edit of 21 frames may take. It took
// 75 seconds while the text encoder ran a prompt's padding with the prompt, 41
// without, 30 with the autoencoder's convolutions computed in tiles, 20 handed
// to its session as it stands, and takes 17.6 with the text encoder
// multiplying by its weights as stored: 11 from a session already loaded, 8.8
// when the instruction is one the encoder has kept. The ceiling stands a
// quarter over the slowest of the three, so a part's cost coming back whole
// fails here. The model's own code takes four seconds over the same clip on
// this device, and each part mended brings the ceiling down.
const servedEditCeiling = 22 * time.Second

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
	width, height, err := latentvideo.EditSourceSize(modeltest.Directory(t, "LiveEdit"))
	if err != nil {
		t.Skipf("UNAVAILABLE: %v", err)
	}
	const frames = 21
	edits := serveEdits(t, "liveedit-stage-times", stripedClip(frames, height, width))
	service := editService(t)
	for _, request := range []struct {
		name, prompt string
		seed         int64
	}{
		// Instructions no other test of this process gives: the encoder keeps
		// the rows of a prompt it has encoded, and the first two requests here
		// are to pay for theirs.
		{"nothing loaded", "turn the stripes green and time each part", 3},
		{"another instruction", "turn the stripes into falling snow and time each part", 3},
		{"another seed", "turn the stripes into falling snow and time each part", 4},
	} {
		ctx, stages := processmeasure.WithStages(t.Context())
		clock := processmeasure.NewStopwatch()
		_, video := edits.serve(ctx, t, service, request.prompt, request.seed)
		elapsed, err := clock.Elapsed()
		if err != nil {
			t.Fatal(err)
		}
		if video.Frames != frames || video.Width != width || video.Height != height {
			t.Fatalf("%s: a served edit is %d frames of %d by %d", request.name, video.Frames, video.Width, video.Height)
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
