package projector

import (
	"bytes"
	"image"
	"image/color"
	"image/gif"
	"testing"
)

func TestDecodeGIFVideoCompositesDisposalAndSamples(t *testing.T) {
	palette := color.Palette{
		color.RGBA{0, 0, 0, 0}, color.RGBA{255, 0, 0, 255},
		color.RGBA{0, 255, 0, 255}, color.RGBA{0, 0, 255, 255},
	}
	full := image.NewPaletted(image.Rect(0, 0, 2, 1), palette)
	full.Pix = []uint8{1, 1}
	green := image.NewPaletted(image.Rect(0, 0, 1, 1), palette)
	green.Pix = []uint8{2}
	blue := image.NewPaletted(image.Rect(1, 0, 2, 1), palette)
	blue.Pix = []uint8{3}
	final := image.NewPaletted(image.Rect(0, 0, 1, 1), palette)
	final.Pix = []uint8{2}
	encoded := gif.GIF{
		Image: []*image.Paletted{full, green, blue, final}, Delay: []int{1, 1, 1, 1},
		Disposal: []byte{gif.DisposalNone, gif.DisposalBackground, gif.DisposalPrevious, gif.DisposalNone},
		Config:   image.Config{ColorModel: palette, Width: 2, Height: 1}, BackgroundIndex: 0,
	}
	var data bytes.Buffer
	if err := gif.EncodeAll(&data, &encoded); err != nil {
		t.Fatal(err)
	}
	frames, err := DecodeGIFVideo(bytes.NewReader(data.Bytes()), 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 4 {
		t.Fatalf("frames = %d", len(frames))
	}
	want := [][2]color.RGBA{
		{{255, 0, 0, 255}, {255, 0, 0, 255}},
		{{0, 255, 0, 255}, {255, 0, 0, 255}},
		{{0, 0, 0, 255}, {0, 0, 255, 255}},
		{{0, 255, 0, 255}, {255, 0, 0, 255}},
	}
	for frameIndex, frame := range frames {
		for x := 0; x < 2; x++ {
			got := color.RGBAModel.Convert(frame.At(x, 0)).(color.RGBA)
			if got != want[frameIndex][x] {
				t.Fatalf("frame %d pixel %d = %v, want %v", frameIndex, x, got, want[frameIndex][x])
			}
		}
	}
	sampled, err := DecodeGIFVideo(bytes.NewReader(data.Bytes()), 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(sampled) != 3 || color.RGBAModel.Convert(sampled[2].At(0, 0)).(color.RGBA) != want[3][0] {
		t.Fatalf("sampled frames = %d", len(sampled))
	}
}

func TestDecodeGIFVideoRejectsInvalidInput(t *testing.T) {
	if _, err := DecodeGIFVideo(bytes.NewReader(nil), 1); err == nil {
		t.Fatal("empty GIF was accepted")
	}
	if _, err := DecodeGIFVideo(bytes.NewReader([]byte("GIF89a")), 0); err == nil {
		t.Fatal("zero frame limit was accepted")
	}
}
