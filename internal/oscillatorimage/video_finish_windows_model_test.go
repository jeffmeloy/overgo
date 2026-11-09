//go:build windows && modeltest

package oscillatorimage

import (
	"os"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestTheOscillatorFinishesTheGIFItIsAskedFor holds the oscillator's video to the finish its request asks, a GIF as an
// MP4 (SQA-359): a GIF asked for at twice its size is finished on the card by the encoder every finished clip is
// written by, every frame at twice the scaled size, and its record names the finish, where the GIF that finishes
// nothing wrote it at the scaled size and recorded none.
func TestTheOscillatorFinishesTheGIFItIsAskedFor(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_NVVFX") == "" {
		testskip.NotApplicable(t, "OVERGO_NVVFX names no NVIDIA Video Effects runtime here")
	}
	const frames, scale = 4, 8
	m := loadArtifactModel(t)
	request := VideoRequest{Class: 1, Seed: 202, Frames: frames, Scale: scale, Format: "gif"}
	request.Upscale = 2
	plan := testcheck.Must(m.prepareVideo(t.Context(), request))(t)
	features := testcheck.Must(m.integrateVideo(t.Context(), plan))(t)
	encoded := testcheck.Must(m.decodeVideo(t.Context(), features))(t)
	finish := media.EncodingOf(encoded).Finish
	if encoded.MediaType != media.GIFMediaType || encoded.Frames != frames || encoded.Height != 2*m.Cfg.OutH()*scale ||
		encoded.Width != 2*m.Cfg.OutW()*scale || finish == nil || finish.Scale != 2 {
		t.Fatalf("the oscillator's GIF asked for at twice its size is %s of %d frames at %dx%d, its finish %+v; want %dx%d",
			encoded.MediaType, encoded.Frames, encoded.Width, encoded.Height, finish, 2*m.Cfg.OutW()*scale, 2*m.Cfg.OutH()*scale)
	}
}
