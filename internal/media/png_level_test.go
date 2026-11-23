package media

import (
	"bytes"
	"encoding/binary"
	"image/png"
	"math"
	"testing"

	"overgo/internal/testcheck"
)

// TestAPictureIsWrittenAtComfyUIsLevel holds png-level/do: a picture encoded from unit pixels, MOSS's, from normalized
// pixels and from normalized planar pixels decodes back to the bytes each pixel quantizes to; MOSS's stream says
// deflate's fast levels (2 to 5, of which ComfyUI's SaveImage writes 4), and the others' its default level, the bytes
// their models' outputs are pinned by (Un-0's frozen pictures among them). image/png's default level 6 had taken about
// half of each MOSS Ultra section's decode part, 211 to 264 ms a 1024 x 576 picture against 55 to 66 ms at level 4.
func TestAPictureIsWrittenAtComfyUIsLevel(t *testing.T) {
	t.Parallel()
	const height, width = 37, 53
	unit, normalized, planar := make([]float32, height*width*RGBChannels), make([]float32, height*width*RGBChannels), make([]float32, height*width*RGBChannels)
	for index := range unit {
		unit[index] = float32(0.5 + 0.55*math.Sin(float64(index)/5.7)*math.Cos(float64(index)/97))
		normalized[index] = 2*unit[index] - 1
	}
	for pixel := range height * width {
		for channel := range RGBChannels {
			planar[channel*height*width+pixel] = normalized[pixel*RGBChannels+channel]
		}
	}
	unitByte := func(value float32) byte {
		return byte(math.RoundToEven(float64(float32(min(max(value, 0), 1) * 255))))
	}
	// The level class each stream declares (RFC 1950 FLEVEL): 1 for deflate's levels 2 to 5, 2 for its default, 6.
	const fast, standard = 1, 2
	for _, picture := range []struct {
		name    string
		encoded func() ([]byte, float32, float32, error)
		want    func(index int) byte
		level   int
	}{
		{"unit", func() ([]byte, float32, float32, error) { return EncodeUnitRGBPNG(unit, height, width) }, func(index int) byte { return unitByte(unit[index]) }, fast},
		{"normalized", func() ([]byte, float32, float32, error) {
			return EncodeNormalizedRGBPNG(normalized, height, width, false)
		},
			func(index int) byte { return normalizedPixel8(normalized[index]) }, standard},
		{"planar", func() ([]byte, float32, float32, error) { return EncodeNormalizedRGBPNG(planar, height, width, true) },
			func(index int) byte { return normalizedPixel8(normalized[index]) }, standard},
	} {
		encoded, _, _, err := picture.encoded()
		testcheck.NoError(t, err)
		decoded := testcheck.Must(png.Decode(bytes.NewReader(encoded)))(t)
		for y := range height {
			for x := range width {
				r, g, b, a := decoded.At(x, y).RGBA()
				index := (y*width + x) * RGBChannels
				if got := [4]byte{byte(r >> 8), byte(g >> 8), byte(b >> 8), byte(a >> 8)}; got != [4]byte{picture.want(index), picture.want(index + 1), picture.want(index + 2), 255} {
					t.Fatalf("the %s picture's pixel at %d,%d decodes to %v", picture.name, x, y, got)
				}
			}
		}
		if level := deflateLevelClass(t, encoded); level != picture.level {
			t.Errorf("the %s picture's stream declares deflate level class %d; want %d", picture.name, level, picture.level)
		}
	}
}

// deflateLevelClass is the compression level class the zlib header of a PNG's first IDAT chunk declares (RFC 1950
// FLEVEL): 0 for level 1, 1 for 2 to 5, 2 for 6 and 3 for 7 to 9.
func deflateLevelClass(t *testing.T, encoded []byte) int {
	t.Helper()
	const signature, header = 8, 8
	for at := signature; at+header <= len(encoded); {
		length := int(binary.BigEndian.Uint32(encoded[at:]))
		if string(encoded[at+4:at+header]) == "IDAT" && length >= 2 {
			return int(encoded[at+header+1] >> 6)
		}
		at += header + length + 4
	}
	t.Fatal("the PNG holds no IDAT chunk")
	return -1
}
