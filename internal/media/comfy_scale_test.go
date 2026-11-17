package media

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"image/png"
	"math"
	"os"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// TestScaleMatchesComfyUIsImageScale holds the scaling of image to image's source (moss-ultra-sections/img2img) to
// ComfyUI's ImageScale at lanczos with a centre crop, run as it is (scripts/python_golden_references/
// comfy_image_scale_golden.py, Pillow 12.3.0): a 160 x 96 crop of the recorder's input scaled to its own size (a copy,
// through ComfyUI's 8-bit round trip), down, up, cropped to a square, to a portrait, to sizes no scale divides and to a
// thin strip, each to the bytes ComfyUI gives. A crop margin that rounds half way rounds to even, as Python's does.
func TestScaleMatchesComfyUIsImageScale(t *testing.T) {
	t.Parallel()
	var golden struct {
		Source string `json:"source_sha256"`
		Cases  []struct {
			Width  int    `json:"width"`
			Height int    `json:"height"`
			RGB    string `json:"rgb_sha256"`
		} `json:"cases"`
		Bilinear []struct {
			Width   int    `json:"width"`
			Height  int    `json:"height"`
			Float32 string `json:"float32_sha256"`
		} `json:"bilinear"`
	}
	testcheck.NoError(t, jsonfile.Decode("testdata/comfy_scale/scale.json", &golden))
	data := testcheck.Must(os.ReadFile("testdata/comfy_scale/source.png"))(t)
	if sum := sha256.Sum256(data); hex.EncodeToString(sum[:]) != golden.Source {
		t.Fatalf("the source hashes %x; want the one the golden scaled, %s", sum, golden.Source)
	}
	file := testcheck.Must(os.Open("testdata/comfy_scale/source.png"))(t)
	defer file.Close()
	picture := testcheck.Must(png.Decode(file))(t)
	if len(golden.Cases) == 0 {
		t.Fatal("the golden scales to no canvas")
	}
	for _, scaled := range golden.Cases {
		levels := testcheck.Must(ScaleLikeComfyUI(picture, scaled.Width, scaled.Height))(t)
		if sum := sha256.Sum256(levels); hex.EncodeToString(sum[:]) != scaled.RGB {
			t.Errorf("scaled to %dx%d, the picture hashes %x; want ComfyUI's %s", scaled.Width, scaled.Height, sum, scaled.RGB)
		}
	}
	// At bilinear without a crop, as the depth section scales its map back, ComfyUI scales the picture's float32 values
	// over 255: up, down and to sizes no scale divides, each to ComfyUI's float32 bytes (moss-ultra-sections/depth).
	bounds := picture.Bounds()
	unit := make([]float32, 0, bounds.Dx()*bounds.Dy()*RGBChannels)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := picture.At(x, y).RGBA()
			unit = append(unit, float32(r>>8)/UnitLevels, float32(g>>8)/UnitLevels, float32(b>>8)/UnitLevels)
		}
	}
	if len(golden.Bilinear) == 0 {
		t.Fatal("the golden scales bilinearly to no canvas")
	}
	for _, scaled := range golden.Bilinear {
		values := testcheck.Must(BilinearLikeComfyUI(unit, bounds.Dy(), bounds.Dx(), RGBChannels, scaled.Height, scaled.Width))(t)
		data := make([]byte, 0, 4*len(values))
		for _, value := range values {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(value))
		}
		if sum := sha256.Sum256(data); hex.EncodeToString(sum[:]) != scaled.Float32 {
			t.Errorf("scaled bilinearly to %dx%d, the picture hashes %x; want ComfyUI's %s", scaled.Width, scaled.Height, sum, scaled.Float32)
		}
	}
	for _, crop := range []struct{ inputWidth, inputHeight, width, height, left, top int }{
		{1024, 576, 1024, 576, 0, 0}, {1024, 576, 576, 576, 224, 0}, {576, 1024, 1024, 576, 0, 350}, {5, 2, 2, 2, 2, 0}, {7, 2, 2, 2, 2, 0},
	} {
		if left, top := centreCrop(crop.inputWidth, crop.inputHeight, crop.width, crop.height); left != crop.left || top != crop.top {
			t.Errorf("a %dx%d picture cropped for %dx%d loses %d and %d a side; want %d and %d", crop.inputWidth, crop.inputHeight, crop.width, crop.height, left, top, crop.left, crop.top)
		}
	}
	if _, err := ScaleLikeComfyUI(picture, 0, 10); err == nil {
		t.Error("a canvas of no width was scaled to; want it refused")
	}
}
