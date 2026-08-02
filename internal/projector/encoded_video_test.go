package projector

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/gif"
	"path/filepath"
	"testing"
)

func TestDecodeEncodedVideoGIF(t *testing.T) {
	var data bytes.Buffer
	frames := &gif.GIF{
		Image: []*image.Paletted{
			image.NewPaletted(image.Rect(0, 0, 2, 2), color.Palette{color.Black, color.White}),
			image.NewPaletted(image.Rect(0, 0, 2, 2), color.Palette{color.Black, color.White}),
		},
		Delay: []int{1, 1},
	}
	if err := gif.EncodeAll(&data, frames); err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeEncodedVideo(context.Background(), data.Bytes(), "", 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != 1 || decoded[0].Bounds() != image.Rect(0, 0, 2, 2) {
		t.Fatalf("decoded frames = %d, bounds=%v", len(decoded), decoded[0].Bounds())
	}
}

func TestResolveFFmpegRejectsMissingConfiguredExecutable(t *testing.T) {
	if _, err := ResolveFFmpeg(filepath.Join(t.TempDir(), "missing.exe")); err == nil {
		t.Fatal("missing FFmpeg executable was accepted")
	}
}
