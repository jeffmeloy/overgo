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

// TestServedEditStageTimes reports cold, changed-instruction and changed-seed stage times.
// Geometry and stage accounting remain assertions; elapsed times are observations.
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
		if unnamed < 0 {
			t.Errorf("%s: no part names %.2f s of a request of %.2f s", request.name, unnamed.Seconds(), wall.Seconds())
		}
	}
}
