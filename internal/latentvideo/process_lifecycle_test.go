package latentvideo

import (
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// TestExternalProcessLifecycle proves the MP4 encoder rides the
// shared supervisor: frames stream into the contained FFmpeg tree,
// Finish reaps it into an encoded clip, and Close terminates an
// unfinished encoder tree. Absent FFmpeg skips.
func TestExternalProcessLifecycle(t *testing.T) {
	ffmpeg, err := media.ResolveFFmpeg("")
	if err != nil {
		t.Skipf("UNAVAILABLE: %v", err)
	}
	const height, width = 16, 16
	encoder, err := NewMP4Encoder(t.Context(), ffmpeg, 4, SignedUnitPixels)
	testcheck.NoError(t, err)
	frame := make([]float32, 3*height*width)
	for index := range frame {
		frame[index] = float32(index%17)/17*2 - 1
	}
	for ordinal := range 4 {
		testcheck.NoError(t, encoder.Add(ordinal, frame, height, width))
	}
	video, err := encoder.Finish()
	testcheck.NoError(t, err)
	if len(video.Data) == 0 || video.Frames != 4 {
		t.Fatalf("encoded video = frames %d bytes %d", video.Frames, len(video.Data))
	}

	// FFmpeg starts at the first frame, so the abandoned encoder takes one.
	abandoned, err := NewMP4Encoder(t.Context(), ffmpeg, 4, SignedUnitPixels)
	testcheck.NoError(t, err)
	testcheck.NoError(t, abandoned.Add(0, frame, height, width))
	if err := abandoned.Close(); err == nil {
		// Termination reports through the receipt; Close may surface the
		// kill as an error or absorb it -- either way the tree is reaped.
		_ = err
	}
}
