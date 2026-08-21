package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/media"
)

func TestDecodeVideoFileFFmpeg(t *testing.T) {
	requireIntegration(t)
	ffmpeg := os.Getenv("OVERGO_FFMPEG_TEST")
	if ffmpeg == "" {
		t.Skip("set OVERGO_FFMPEG_TEST to run FFmpeg integration tests")
	}
	path := filepath.Join(t.TempDir(), "input.mkv")
	command := exec.Command(
		ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
		"-i", "testsrc=size=16x16:rate=4:duration=1", "-c:v", "ffv1", path,
	)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("create test video: %v: %s", err, output)
	}
	frames, err := media.DecodeVideoFile(context.Background(), path, ffmpeg, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 2 {
		t.Fatalf("decoded frames = %d, want 2", len(frames))
	}
	for index, frame := range frames {
		if frame.Bounds().Dx() != 16 || frame.Bounds().Dy() != 16 {
			t.Fatalf("frame %d bounds = %v", index, frame.Bounds())
		}
	}
}

func TestResolveFFmpegRejectsMissingConfiguredPath(t *testing.T) {
	if _, err := media.ResolveFFmpeg(filepath.Join(t.TempDir(), "missing.exe")); err == nil {
		t.Fatal("missing configured FFmpeg was accepted")
	}
}
