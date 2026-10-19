package latentimage

import (
	"bytes"
	"encoding/binary"
	"image/png"
	"io"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/safetensors"
)

func mossPixelGolden(t *testing.T) {
	mossFixtureBytes(t, "pixels.safetensors", "9a21e4e43ca18c249355898f426e8bd9db0176fd24812505ca0b66828c4a993b")
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "pixels.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, name := range []string{"square", "wide", "tall", "ties"} {
		t.Run(name, func(t *testing.T) {
			key := name + ".pixels"
			if name == "ties" {
				key = name + ".input"
			}
			shape := s.Tensors[key].Shape
			h, w := int(shape[1]), int(shape[2])
			values := refinementValues(t, s, key)
			got, err := encodeUnitPNG(values, h, w)
			if err != nil {
				t.Fatal(err)
			}
			if got.Height != h || got.Width != w || got.Channels != 3 || got.MediaType != "image/png" {
				t.Fatalf("encoded geometry differs: %+v", got)
			}
			repeat, err := encodeUnitPNG(values, h, w)
			if err != nil || !bytes.Equal(got.Data, repeat.Data) {
				t.Fatalf("encoded pixels do not repeat: %v", err)
			}
			content, err := PNGContent(got)
			if err != nil {
				t.Fatal(err)
			}
			if err = content.Validate(); err != nil {
				t.Fatal(err)
			}
			decoded, err := png.Decode(bytes.NewReader(got.Data))
			if err != nil {
				t.Fatal(err)
			}
			if decoded.Bounds().Dx() != w || decoded.Bounds().Dy() != h {
				t.Fatal("PNG geometry differs")
			}
			raw, err := io.ReadAll(s.Tensors[name+".rgb"].Reader())
			if err != nil {
				t.Fatal(err)
			}
			if len(raw) != h*w*3*8 {
				t.Fatal("native RGB fixture length differs")
			}
			for y := range h {
				for x := range w {
					r, g, b, a := decoded.At(x, y).RGBA()
					if a != 65535 {
						t.Fatal("encoded pixel is not opaque")
					}
					for c, v := range []uint32{r >> 8, g >> 8, b >> 8} {
						i := (y*w+x)*3 + c
						want := binary.LittleEndian.Uint64(raw[8*i:])
						if uint64(v) != want {
							t.Fatalf("RGB channel %d differs: got %d, native %d", i, v, want)
						}
					}
				}
			}
			t.Logf("%d native RGB channels match exactly", len(values))
		})
	}
}

func TestUnitPNGRejectsInvalidPixels(t *testing.T) {
	for _, tc := range []struct {
		name   string
		pixels []float32
		h, w   int
	}{
		{"empty", nil, 1, 1},
		{"short", []float32{0, 1}, 1, 1},
		{"zero", nil, 0, 1},
		{"negative", nil, 1, -1},
		{"nan", []float32{0, float32(math.NaN()), 1}, 1, 1},
		{"inf", []float32{0, float32(math.Inf(1)), 1}, 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := encodeUnitPNG(tc.pixels, tc.h, tc.w); err == nil {
				t.Fatal("invalid image accepted")
			}
		})
	}
}
