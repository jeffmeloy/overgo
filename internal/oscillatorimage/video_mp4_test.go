package oscillatorimage

import (
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestTheOscillatorWritesTheMP4ItIsAskedFor holds the oscillator's video to the format its request names: a request
// the intake gives MP4 is written through the encoder Wan's clips are, an MP4 of every frame at the scaled size that
// names the FFmpeg that wrote it, where a request naming none keeps its GIF (TestArtifactPublishesReferenceGIF). A
// format neither GIF nor MP4 is refused with the request.
func TestTheOscillatorWritesTheMP4ItIsAskedFor(t *testing.T) {
	t.Parallel()
	const frames, scale = 6, 8
	if ValidateVideoRequest(VideoRequest{Class: 1, Frames: frames, Scale: scale, Format: "webm"}) == nil {
		t.Fatal("a request for a WebM clip was admitted")
	}
	m := loadArtifactModel(t)
	if _, err := media.FFmpeg(); err != nil {
		testskip.NotApplicable(t, "no FFmpeg to write an MP4 with: "+err.Error())
	}
	plan := testcheck.Must(m.prepareVideo(t.Context(), VideoRequest{Class: 1, Seed: 202, Frames: frames, Scale: scale, Format: "mp4"}))(t)
	features := testcheck.Must(m.integrateVideo(t.Context(), plan))(t)
	encoded := testcheck.Must(m.decodeVideo(t.Context(), features))(t)
	if encoded.MediaType != media.MP4MediaType || !media.OpensAsMP4(encoded.Data) || encoded.Frames != frames ||
		encoded.Height != m.Cfg.OutH()*scale || encoded.Width != m.Cfg.OutW()*scale || media.EncodingOf(encoded).IsZero() {
		t.Fatalf("the oscillator's MP4 is %s of %d frames at %dx%d, written by %+v", encoded.MediaType, encoded.Frames, encoded.Width, encoded.Height, encoded.Encoding)
	}
}
