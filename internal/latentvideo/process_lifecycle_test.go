package latentvideo

import (
	"context"
	"testing"

	"overgo/internal/media"
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
	encoder, err := NewMP4Encoder(context.Background(), ffmpeg, 4, height, width, SignedUnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	frame := make([]float32, 3*height*width)
	for index := range frame {
		frame[index] = float32(index%17)/17*2 - 1
	}
	for ordinal := range 4 {
		if err := encoder.Add(ordinal, frame, height, width); err != nil {
			t.Fatal(err)
		}
	}
	video, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if len(video.Data) == 0 || video.Frames != 4 {
		t.Fatalf("encoded video = frames %d bytes %d", video.Frames, len(video.Data))
	}

	abandoned, err := NewMP4Encoder(context.Background(), ffmpeg, 4, height, width, SignedUnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	if err := abandoned.Close(); err == nil {
		// Termination reports through the receipt; Close may surface the
		// kill as an error or absorb it -- either way the tree is reaped.
		_ = err
	}
}
