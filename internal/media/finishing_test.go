package media

import (
	"image/png"
	"math"
	"os"
	"strconv"
	"testing"

	"overgo/internal/checked"
	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// comfyFinishing is ComfyUI's ImageSharpen and ResolutionSelector run as they are
// (scripts/python_golden_references/comfy_finishing_golden.py): the node's sharpening of a crop of the Ultra workflow's
// recorded original, and its sizes over every aspect ratio, a spread of megapixels and multiples.
type comfyFinishing struct {
	Sharpen struct {
		Radius int     `json:"sharpen_radius"`
		Sigma  float64 `json:"sigma"`
		Alpha  float64 `json:"alpha"`
	} `json:"sharpen"`
	Sizes []struct {
		Ratio      [2]int  `json:"ratio"`
		Megapixels float64 `json:"megapixels"`
		Multiple   int     `json:"multiple"`
		Width      int     `json:"width"`
		Height     int     `json:"height"`
	} `json:"sizes"`
}

// recordedLevels is a recorded PNG's 8-bit RGB levels, interleaved.
func recordedLevels(t *testing.T, path string) ([]uint8, int, int) {
	t.Helper()
	file := testcheck.Must(os.Open(path))(t)
	defer file.Close()
	picture := testcheck.Must(png.Decode(file))(t)
	bounds := picture.Bounds()
	levels := make([]uint8, 0, bounds.Dx()*bounds.Dy()*RGBChannels)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := picture.At(x, y).RGBA()
			levels = append(levels, RGBAChannel8(r), RGBAChannel8(g), RGBAChannel8(b))
		}
	}
	return levels, bounds.Dy(), bounds.Dx()
}

// savedLevel is a unit value as ComfyUI's SaveImage writes it: the floor of 255 times it, clipped.
func savedLevel(value float32) int {
	return min(max(int(math.Floor(float64(value)*255)), 0), 255)
}

// TestFinishingMatchesTheRecordedImage holds the MOSS Ultra workflow's finishing (moss-ultra-sections/finishing) to
// ComfyUI's own nodes: overgo's sharpening, at the workflow's radius 1, sigma 0.6 and alpha 0.7, of a crop of the
// recorded original, read as LoadImage reads a picture, is within one 8-bit level of the node's sharpening of the same
// crop as SaveImage saves it; and every size ResolutionSelector gave, eight aspect ratios over eleven megapixels and
// five multiples, the workflow's 1024 × 576 among them, is the one overgo sizes.
func TestFinishingMatchesTheRecordedImage(t *testing.T) {
	var recorded comfyFinishing
	testcheck.NoError(t, jsonfile.Decode("testdata/comfy_finishing/finishing.json", &recorded))
	original, height, width := recordedLevels(t, "testdata/comfy_finishing/original_crop.png")
	want, _, _ := recordedLevels(t, "testdata/comfy_finishing/sharpened_crop.png")
	pixels := make([]float32, len(original))
	for i, level := range original {
		pixels[i] = float32(level) / 255
	}
	sharpened := testcheck.Must(Sharpen(pixels, height, width, Sharpening{Radius: recorded.Sharpen.Radius, Sigma: recorded.Sharpen.Sigma, Alpha: recorded.Sharpen.Alpha}))(t)
	exact, worst := 0, 0
	for i, value := range sharpened {
		difference := savedLevel(value) - int(want[i])
		worst = max(worst, difference, -difference)
		if difference == 0 {
			exact++
		}
	}
	t.Logf("sharpened %d×%d: %d of %d values exact, the worst %d levels off", width, height, exact, len(sharpened), worst)
	if worst > 1 {
		t.Errorf("the sharpened crop is %d levels off ComfyUI's at worst, want at most 1", worst)
	}
	sized := map[[2]int]bool{}
	for _, size := range recorded.Sizes {
		canvas := Canvas{Aspect: strconv.Itoa(size.Ratio[0]) + ":" + strconv.Itoa(size.Ratio[1]), Megapixels: size.Megapixels, Multiple: size.Multiple}
		width, height, err := canvas.Size()
		if err != nil || width != size.Width || height != size.Height {
			t.Errorf("%d:%d at %g megapixels in multiples of %d: %d×%d (%v), ComfyUI's %d×%d", size.Ratio[0], size.Ratio[1], size.Megapixels, size.Multiple, width, height, err, size.Width, size.Height)
		}
		sized[[2]int{width, height}] = true
	}
	if !sized[[2]int{1024, 576}] || len(recorded.Sizes) < 400 {
		t.Errorf("the recording holds %d sizes and the workflow's 1024×576 %v", len(recorded.Sizes), sized[[2]int{1024, 576}])
	}
}

// TestEverySharpeningAdmittedDrawsAFinitePicture holds every sharpening the request's validation admits before a model
// opens (Sharpening.Validate, any positive sigma, any alpha up to the largest whose kernel sums finitely) to a finite
// picture (SQA-393): the Gaussian's weights divided by 2·sigma², which underflows to 0 below a sigma of about 1e-154, so
// the kernel's centre was 0/0 and every value NaN (sharpen-admit/do); and an alpha of 1e308 was admitted, its taps
// overflowed and the picture was NaN, drawn whole and then lost (sharpen-admit/alpha). The test was named
// TestSharpeningIsAdmittedBeforeTheModelOpens.
func TestEverySharpeningAdmittedDrawsAFinitePicture(t *testing.T) {
	pixels := make([]float32, 4*4*RGBChannels)
	for i := range pixels {
		pixels[i] = float32(i%7) / 7
	}
	for _, sigma := range []float64{1e-200, 1e-160, 0.6, 1e200} {
		for _, alpha := range []float64{0.7, math.MaxFloat64 / 20} {
			sharpening := Sharpening{Radius: 1, Sigma: sigma, Alpha: alpha}
			testcheck.NoError(t, sharpening.Validate())
			for i, value := range testcheck.Must(Sharpen(pixels, 4, 4, sharpening))(t) {
				if !checked.Finite32(value) {
					t.Fatalf("sharpening at sigma %g and alpha %g gave value %d as %g", sigma, alpha, i, value)
				}
			}
		}
	}
	// An alpha whose taps overflow is refused before the model opens (SQA-393, its alpha remainder).
	if err := (Sharpening{Radius: 1, Sigma: 0.6, Alpha: 1e308}).Admit(4, 4); err == nil {
		t.Error("an alpha of 1e308, whose taps overflow, was admitted")
	}
}

// TestSharpenRefusesWhatItCannotSharpen holds the sharpening's bounds.
func TestSharpenRefusesWhatItCannotSharpen(t *testing.T) {
	pixels := make([]float32, 4*4*RGBChannels)
	for _, bad := range []struct {
		height, width, radius int
		sigma, alpha          float64
	}{{4, 4, 0, 1, 1}, {4, 4, 4, 1, 1}, {4, 4, 1, 0, 1}, {4, 4, 1, 1, -1}, {4, 5, 1, 1, 1}} {
		if _, err := Sharpen(pixels, bad.height, bad.width, Sharpening{Radius: bad.radius, Sigma: bad.sigma, Alpha: bad.alpha}); err == nil {
			t.Errorf("sharpened %+v", bad)
		}
	}
	for _, canvas := range []Canvas{{Aspect: "16:9", Megapixels: 0.0001, Multiple: 128}, {Aspect: "wide", Megapixels: 1, Multiple: 32}, {Aspect: "16:0", Megapixels: 1, Multiple: 32}} {
		if _, _, err := canvas.Size(); err == nil {
			t.Errorf("sized %+v", canvas)
		}
	}
}
