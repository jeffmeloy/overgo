package media

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/processcontrol"
)

// TestExternalProcessLifecycle proves media decoding rides the shared
// supervisor: a synthesized clip decodes to frames through the
// contained FFmpeg tree. Absent FFmpeg skips: the media path is
// exercised wherever its dependency exists.
func TestExternalProcessLifecycle(t *testing.T) {
	ffmpeg, err := ResolveFFmpeg("")
	if err != nil {
		t.Skipf("UNAVAILABLE: %v", err)
	}
	clip := filepath.Join(t.TempDir(), "lifecycle.mp4")
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path: ffmpeg,
		Args: []string{"-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
			"testsrc=size=64x64:rate=4:duration=1", "-pix_fmt", "yuv420p", clip},
	})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("synthesize clip = (%+v, %v)", receipt, err)
	}
	frames, err := decodeFFmpegFrames(context.Background(), ffmpeg, clip, nil, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) == 0 {
		t.Fatal("decoded no frames")
	}
}
