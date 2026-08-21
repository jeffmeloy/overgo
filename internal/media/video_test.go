package media

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/gif"
	"path/filepath"
	"testing"

	"overgo/internal/tensor"
)

func TestDecodeGIFCompositesDisposalAndSamples(t *testing.T) {
	palette := color.Palette{
		color.RGBA{}, color.RGBA{R: ^uint8(0), A: ^uint8(0)},
		color.RGBA{G: ^uint8(0), A: ^uint8(0)}, color.RGBA{B: ^uint8(0), A: ^uint8(0)},
	}
	full := image.NewPaletted(image.Rect(0, 0, tensor.PairedExtent, tensor.SingletonExtent), palette)
	full.Pix = []uint8{tensor.SingletonExtent, tensor.SingletonExtent}
	green := image.NewPaletted(image.Rect(0, 0, tensor.SingletonExtent, tensor.SingletonExtent), palette)
	green.Pix = []uint8{tensor.PairedExtent}
	blue := image.NewPaletted(image.Rect(tensor.SingletonExtent, 0, tensor.PairedExtent, tensor.SingletonExtent), palette)
	blue.Pix = []uint8{RGBChannels}
	final := image.NewPaletted(image.Rect(0, 0, tensor.SingletonExtent, tensor.SingletonExtent), palette)
	final.Pix = []uint8{tensor.PairedExtent}
	encoded := gif.GIF{
		Image:    []*image.Paletted{full, green, blue, final},
		Delay:    []int{tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent},
		Disposal: []byte{gif.DisposalNone, gif.DisposalBackground, gif.DisposalPrevious, gif.DisposalNone},
		Config:   image.Config{ColorModel: palette, Width: tensor.PairedExtent, Height: tensor.SingletonExtent},
	}
	var data bytes.Buffer
	if err := gif.EncodeAll(&data, &encoded); err != nil {
		t.Fatal(err)
	}
	frames, err := DecodeGIF(bytes.NewReader(data.Bytes()), len(encoded.Image))
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != len(encoded.Image) {
		t.Fatalf("frames = %d", len(frames))
	}
	want := [][tensor.PairedExtent]color.RGBA{
		{{R: ^uint8(0), A: ^uint8(0)}, {R: ^uint8(0), A: ^uint8(0)}},
		{{G: ^uint8(0), A: ^uint8(0)}, {R: ^uint8(0), A: ^uint8(0)}},
		{{A: ^uint8(0)}, {B: ^uint8(0), A: ^uint8(0)}},
		{{G: ^uint8(0), A: ^uint8(0)}, {R: ^uint8(0), A: ^uint8(0)}},
	}
	for frameIndex, frame := range frames {
		for x := range tensor.PairedExtent {
			got := color.RGBAModel.Convert(frame.At(x, tensor.FirstOffset)).(color.RGBA)
			if got != want[frameIndex][x] {
				t.Fatalf("frame %d pixel %d = %v, want %v", frameIndex, x, got, want[frameIndex][x])
			}
		}
	}
	sampled, err := DecodeGIF(bytes.NewReader(data.Bytes()), RGBChannels)
	if err != nil {
		t.Fatal(err)
	}
	if len(sampled) != RGBChannels || color.RGBAModel.Convert(sampled[tensor.PairedExtent].At(0, 0)).(color.RGBA) != want[len(want)-tensor.SingletonExtent][tensor.FirstOffset] {
		t.Fatalf("sampled frames = %d", len(sampled))
	}
}

func TestDecodeEncodedVideoGIF(t *testing.T) {
	var data bytes.Buffer
	frames := &gif.GIF{Image: []*image.Paletted{
		image.NewPaletted(image.Rect(0, 0, tensor.PairedExtent, tensor.PairedExtent), color.Palette{color.Black, color.White}),
		image.NewPaletted(image.Rect(0, 0, tensor.PairedExtent, tensor.PairedExtent), color.Palette{color.Black, color.White}),
	}, Delay: []int{tensor.SingletonExtent, tensor.SingletonExtent}}
	if err := gif.EncodeAll(&data, frames); err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeEncodedVideo(context.Background(), data.Bytes(), "", tensor.PairedExtent, tensor.SingletonExtent)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != tensor.SingletonExtent || decoded[tensor.FirstOffset].Bounds() != image.Rect(0, 0, tensor.PairedExtent, tensor.PairedExtent) {
		t.Fatalf("decoded frames = %d, bounds=%v", len(decoded), decoded[tensor.FirstOffset].Bounds())
	}
}

func TestVideoDecodeRejectsInvalidInput(t *testing.T) {
	if _, err := DecodeGIF(bytes.NewReader(nil), tensor.SingletonExtent); err == nil {
		t.Fatal("empty GIF was accepted")
	}
	if _, err := DecodeGIF(bytes.NewReader([]byte("GIF89a")), tensor.FirstOffset); err == nil {
		t.Fatal("zero frame limit was accepted")
	}
	if _, err := ResolveFFmpeg(filepath.Join(t.TempDir(), "missing.exe")); err == nil {
		t.Fatal("missing FFmpeg executable was accepted")
	}
}
