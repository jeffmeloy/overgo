//go:build windows && modeltest

package diffusionimage

import (
	"bytes"
	"context"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

func TestRealResidentGenerationLeadership(t *testing.T) {
	cudatest.Require(t)
	const seed, steps, size = int64(7), 2, 64
	model := loadArtifactModel(t)
	hostDurations := make([]time.Duration, 3)
	var hostPixels []float32
	for index := range hostDurations {
		started := time.Now()
		var err error
		hostPixels, err = model.Sample(1, size, size, steps, seed)
		if err != nil {
			t.Fatal(err)
		}
		hostDurations[index] = time.Since(started)
	}
	forward, err := CompileResidentForward(t.Context(), model, 0, size, size)
	if err != nil {
		t.Fatal(err)
	}
	defer forward.Close(context.Background())
	if _, err := forward.Sample(t.Context(), steps, seed); err != nil {
		t.Fatal(err)
	}
	warmDurations := make([]time.Duration, 3)
	var devicePixels []float32
	for index := range warmDurations {
		started := time.Now()
		devicePixels, err = forward.Sample(t.Context(), steps, seed)
		if err != nil {
			t.Fatal(err)
		}
		warmDurations[index] = time.Since(started)
	}
	slices.Sort(hostDurations)
	slices.Sort(warmDurations)
	hostMedian, warmMedian := hostDurations[1], warmDurations[1]
	pixelDiff := testutil.MaxAbsDiff(devicePixels, hostPixels)
	encoded, err := model.decode(sampleFeatures{pixels: devicePixels, height: size, width: size})
	if err != nil {
		t.Fatal(err)
	}
	referencePath := filepath.Join(filepath.Dir(filepath.Dir(artifactDir(t))), "docs", "image_gen_samples", "uvit_direct_seed7_steps2_64.png")
	reference, err := os.ReadFile(referencePath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: reference image %s: %v", referencePath, err)
	}
	exactPNG := bytes.Equal(encoded.Data, reference)
	pixelChanges, byteMaximum, byteMAE := comparePNGPixels(t, encoded.Data, reference)
	t.Logf("real SimpleDiffusion resident generation: warm_median=%s host_median=%s float_max=%.3e png_bytes=%d exact=%t changed_channels=%d byte_max=%d byte_mae=%.6f", warmMedian, hostMedian, pixelDiff, len(encoded.Data), exactPNG, pixelChanges, byteMaximum, byteMAE)
	if pixelDiff > 3e-5 || byteMaximum > 1 || byteMAE > 0.001 {
		t.Fatalf("resident generation quality differs: float_max=%.3e byte_max=%d byte_mae=%.6f", pixelDiff, byteMaximum, byteMAE)
	}
	if warmMedian*4 >= hostMedian {
		t.Fatalf("resident generation %s does not beat host %s by 4x", warmMedian, hostMedian)
	}
}

func comparePNGPixels(t *testing.T, candidate, reference []byte) (changed, maximum int, mae float64) {
	t.Helper()
	decode := func(data []byte) []uint8 {
		image, err := png.Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		bounds := image.Bounds()
		pixels := make([]uint8, 0, bounds.Dx()*bounds.Dy()*3)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				value := color.NRGBAModel.Convert(image.At(x, y)).(color.NRGBA)
				pixels = append(pixels, value.R, value.G, value.B)
			}
		}
		return pixels
	}
	left, right := decode(candidate), decode(reference)
	if len(left) != len(right) {
		t.Fatalf("PNG pixel counts differ: %d != %d", len(left), len(right))
	}
	var total int
	for index := range left {
		difference := int(left[index]) - int(right[index])
		if difference < 0 {
			difference = -difference
		}
		if difference != 0 {
			changed++
		}
		maximum = max(maximum, difference)
		total += difference
	}
	return changed, maximum, float64(total) / float64(len(left))
}
