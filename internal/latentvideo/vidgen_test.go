package latentvideo

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// TestListCaptionedClips pins the corpus pairing contract: clips pair
// with their manifest captions in deterministic name order, a clip
// without a caption is skipped rather than fabricated, and the limit
// bounds the selection.
func TestListCaptionedClips(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"clip-b.mp4", "clip-a.mp4", "clip-uncaptioned.mp4"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, name), []byte("x"), 0o600))
	}
	manifest := `[{"vid":"clip-a","caption":"a red flower"},{"vid":"clip-b","caption":"a blue door"},{"vid":"absent-clip","caption":"never on disk"}]`
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, vidGenCaptionFile), []byte(manifest), 0o600))
	clips := testcheck.Must(ListCaptionedClips(root, 8))(t)
	if len(clips) != 2 || clips[0].Name != "clip-a" || clips[0].Caption != "a red flower" ||
		clips[1].Name != "clip-b" || clips[1].Caption != "a blue door" {
		t.Fatalf("clips = %+v, want the two captioned clips in name order", clips)
	}
	limited := testcheck.Must(ListCaptionedClips(root, 1))(t)
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
	ctx := t.Context()
	const frames, height, width = 5, 32, 48
	encoder := testcheck.Must(NewMP4Encoder(ctx, ffmpeg, 8, UnitPixels))(t)
	levels := [3]float32{0.8, 0.4, 0.2}
	frame := make([]float32, 3*height*width)
	for channel := range 3 {
		for pixel := range height * width {
			frame[channel*height*width+pixel] = levels[channel]
		}
	}
	for index := range frames {
		testcheck.NoError(t, encoder.Add(index, frame, height, width))
	}
	encoded := testcheck.Must(encoder.Finish())(t)
	path := filepath.Join(t.TempDir(), "solid.mp4")
	testcheck.NoError(t, os.WriteFile(path, encoded.Data, 0o600))
	decoded := testcheck.Must(DecodeClipSource(ctx, ffmpeg, path, SourceVideoShape{Channels: 3, Frames: frames, Height: height, Width: width}))(t)
	spatial := height * width
	for channel := range 3 {
		var sum float64
		for f := range frames {
			for pixel := range spatial {
				sum += float64(decoded[(channel*frames+f)*spatial+pixel])
			}
		}
		mean := sum / float64(frames*spatial)
		if math.Abs(mean-float64(levels[channel])) > 0.05 {
			t.Fatalf("channel %d mean %.4f, want %.2f within codec tolerance", channel, mean, levels[channel])
		}
	}
}
