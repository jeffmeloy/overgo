package latentvideo

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/media"
)

// TestListCaptionedClips pins the corpus pairing contract: clips pair
// with their manifest captions in deterministic name order, a clip
// without a caption is skipped rather than fabricated, and the limit
// bounds the selection.
func TestListCaptionedClips(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"clip-b.mp4", "clip-a.mp4", "clip-uncaptioned.mp4"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	manifest := `[{"vid":"clip-a","caption":"a red flower"},{"vid":"clip-b","caption":"a blue door"},{"vid":"absent-clip","caption":"never on disk"}]`
	if err := os.WriteFile(filepath.Join(root, vidGenCaptionFile), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	clips, err := ListCaptionedClips(root, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(clips) != 2 || clips[0].Name != "clip-a" || clips[0].Caption != "a red flower" ||
		clips[1].Name != "clip-b" || clips[1].Caption != "a blue door" {
		t.Fatalf("clips = %+v, want the two captioned clips in name order", clips)
	}
	limited, err := ListCaptionedClips(root, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(limited) != 1 || limited[0].Name != "clip-a" {
		t.Fatalf("limited = %+v", limited)
	}
}

// TestDecodeClipSource pins the decode convention round-trip: solid
// per-channel frames encoded to MP4 come back as planar [channel]
// [frame][height][width] values in [0, 1] within codec tolerance.
func TestDecodeClipSource(t *testing.T) {
	ffmpeg, err := media.ResolveFFmpeg("")
	if err != nil {
		t.Skip("FFmpeg is unavailable on this host")
	}
	ctx := context.Background()
	const frames, height, width = 5, 32, 48
	encoder, err := NewMP4Encoder(ctx, ffmpeg, 8, height, width, UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	levels := [3]float32{0.8, 0.4, 0.2}
	frame := make([]float32, 3*height*width)
	for channel := 0; channel < 3; channel++ {
		for pixel := 0; pixel < height*width; pixel++ {
			frame[channel*height*width+pixel] = levels[channel]
		}
	}
	for index := 0; index < frames; index++ {
		if err := encoder.Add(index, frame, height, width); err != nil {
			t.Fatal(err)
		}
	}
	encoded, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "solid.mp4")
	if err := os.WriteFile(path, encoded.Data, 0o600); err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeClipSource(ctx, ffmpeg, path, SourceVideoShape{Channels: 3, Frames: frames, Height: height, Width: width})
	if err != nil {
		t.Fatal(err)
	}
	spatial := height * width
	for channel := 0; channel < 3; channel++ {
		var sum float64
		for f := 0; f < frames; f++ {
			for pixel := 0; pixel < spatial; pixel++ {
				sum += float64(decoded[(channel*frames+f)*spatial+pixel])
			}
		}
		mean := sum / float64(frames*spatial)
		if math.Abs(mean-float64(levels[channel])) > 0.05 {
			t.Fatalf("channel %d mean %.4f, want %.2f within codec tolerance", channel, mean, levels[channel])
		}
	}
}
