package media

import (
	"math"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// TestOpenCVResizesMatchCV2 holds the OpenCV resizes the depth preprocessor runs (moss-ultra-sections/depth) to
// cv2.resize itself (scripts/python_golden_references/opencv_resize_golden.py, OpenCV 5.0.0) on a 37 x 23 picture of
// fixed pattern: bicubic on float64 values down, up and to its own size within 1e-12 of OpenCV's; bicubic on 8-bit
// levels up and area on 8-bit levels down, to sizes no scale divides and one it nearly does, to OpenCV's levels exactly.
func TestOpenCVResizesMatchCV2(t *testing.T) {
	t.Parallel()
	var golden struct {
		Height, Width int
		Levels        []uint8
		Cases         []struct {
			Kind          string
			Width, Height int
			Values        []float64
		}
	}
	testcheck.NoError(t, jsonfile.Decode("testdata/opencv_resize/resize.json", &golden))
	values := make([]float64, len(golden.Levels))
	for index, level := range golden.Levels {
		values[index] = float64(level) / 255
	}
	kinds := map[string]int{}
	for _, resized := range golden.Cases {
		kinds[resized.Kind]++
		var got []float64
		switch resized.Kind {
		case "cubic float64":
			got = testcheck.Must(CubicResizeFloat64(values, golden.Height, golden.Width, 3, resized.Height, resized.Width))(t)
		case "cubic levels", "area levels":
			resize := CubicResizeLevels
			if resized.Kind == "area levels" {
				resize = AreaResizeLevels
			}
			levels := testcheck.Must(resize(golden.Levels, golden.Height, golden.Width, 3, resized.Height, resized.Width))(t)
			for _, level := range levels {
				got = append(got, float64(level))
			}
		default:
			t.Fatalf("the golden holds a resize of kind %q", resized.Kind)
		}
		if len(got) != len(resized.Values) {
			t.Fatalf("%s to %dx%d gave %d values; want %d", resized.Kind, resized.Width, resized.Height, len(got), len(resized.Values))
		}
		worst, differ := 0.0, 0
		for index, want := range resized.Values {
			difference := math.Abs(got[index] - want)
			worst = max(worst, difference)
			if difference > 1e-12 {
				differ++
			}
		}
		if differ > 0 {
			t.Errorf("%s to %dx%d: %d of %d values differ from OpenCV's, the worst by %g", resized.Kind, resized.Width, resized.Height, differ, len(got), worst)
		}
	}
	if kinds["cubic float64"] == 0 || kinds["cubic levels"] == 0 || kinds["area levels"] == 0 {
		t.Fatalf("the golden resizes %v; want each kind", kinds)
	}
}
