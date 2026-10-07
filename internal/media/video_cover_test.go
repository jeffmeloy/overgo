package media

import (
	"image"
	"path/filepath"
	"testing"

	"overgo/internal/processcontrol"
)

// TestACoveringCutKeepsAFramesProportions holds the two sized cuts apart on a
// wide clip whose left quarter is red and the rest blue: scaled to a square
// as it is, the square is squashed and still a quarter red; scaled until it
// covers the square and cropped, the square is the clip's middle, which is
// all blue. Absent FFmpeg skips.
func TestACoveringCutKeepsAFramesProportions(t *testing.T) {
	ffmpeg, err := ResolveFFmpeg("")
	if err != nil {
		t.Skipf("UNAVAILABLE: %v", err)
	}
	// 160 by 40: the left 40 columns red, the rest blue.
	clip := filepath.Join(t.TempDir(), "wide.mp4")
	receipt, err := processcontrol.Run(t.Context(), processcontrol.Command{
		Path: ffmpeg,
		Args: []string{"-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
			"color=c=blue:size=160x40:rate=4:duration=1,drawbox=x=0:y=0:w=40:h=40:color=red:t=fill", "-pix_fmt", "yuv444p", clip},
	})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("synthesize clip = (%+v, %v)", receipt, err)
	}
	redShare := func(decode func() ([]image.Image, error)) float64 {
		t.Helper()
		frames, err := decode()
		if err != nil || len(frames) != 1 {
			t.Fatalf("decoded %d frames: %v", len(frames), err)
		}
		bounds := frames[0].Bounds()
		if bounds.Dx() != 40 || bounds.Dy() != 40 {
			t.Fatalf("the frame is %v, want 40 by 40", bounds)
		}
		red := 0
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, _, b, _ := frames[0].At(x, y).RGBA()
				if r > b {
					red++
				}
			}
		}
		return float64(red) / float64(bounds.Dx()*bounds.Dy())
	}
	squashed := redShare(func() ([]image.Image, error) { return DecodeVideoFileAt(t.Context(), clip, ffmpeg, 4, 1, 40, 40) })
	covered := redShare(func() ([]image.Image, error) { return DecodeVideoFileCovering(t.Context(), clip, ffmpeg, 4, 1, 40, 40) })
	// Squashed, the red quarter of the clip is a quarter of the square; the
	// covering cut takes the clip's middle 40 columns, which are all blue.
	if squashed < 0.2 || squashed > 0.3 || covered != 0 {
		t.Fatalf("red is %.2f of the squashed frame and %.2f of the covering one, want a quarter and none", squashed, covered)
	}
	if _, err := DecodeVideoFileCovering(t.Context(), clip, ffmpeg, 4, 1, 0, 40); err == nil {
		t.Fatal("a cut with no width was decoded")
	}
}
