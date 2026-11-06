//go:build windows

package latentimage

import (
	"math"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
)

// encoderTolerance bounds the device's F32 encoder against the CPU's: two
// orderings of the same convolutions and attention, relative to the largest value.
const encoderTolerance = 1e-3

// interleaved turns [channels, height, width] planes into [height*width, channels] rows.
func interleaved(planar []float32, channels int) []float32 {
	out := make([]float32, len(planar))
	plane := len(planar) / channels
	for position := range plane {
		for channel := range channels {
			out[position*channels+channel] = planar[channel*plane+position]
		}
	}
	return out
}

func encoderGoldenCheck(t *testing.T, root, fixture, digest string, height, width, latent int) {
	t.Helper()
	mossFixtureBytes(t, fixture, digest)
	source := testcheck.Must(safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", fixture)))(t)
	defer source.Close()
	read := func(name string) []float32 {
		t.Helper()
		values := testcheck.Must(safetensors.ReadF32(source.Tensors[name]))(t)
		return values
	}
	encoder := testcheck.Must(NewPictureEncoder(t.Context(), root, height, width))(t)
	defer func() {
		if err := encoder.Close(t.Context()); err != nil {
			t.Error(err)
		}
	}()
	pixels := interleaved(read("image"), 3)
	compare := func(name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s holds %d values, want %d", name, len(got), len(want))
		}
		var largest, worst float64
		for index, value := range want {
			largest = max(largest, math.Abs(float64(value)))
			difference := math.Abs(float64(got[index]) - float64(value))
			if !(difference >= 0) {
				t.Fatalf("%s[%d] is %v", name, index, got[index])
			}
			worst = max(worst, difference)
		}
		t.Logf("%s worst %g of largest %g", name, worst, largest)
		if worst > encoderTolerance*largest {
			t.Fatalf("%s differs by %g of largest %g", name, worst, largest)
		}
	}
	mean := testcheck.Must(encoder.mean(t.Context(), pixels))(t)
	compare("moments mean", mean, interleaved(read("moments_mean"), latent))
	rows, gridHeight, gridWidth, err := encoder.Encode(t.Context(), pixels)
	testcheck.NoError(t, err)
	if gridHeight != height/16 || gridWidth != width/16 {
		t.Fatalf("grid %dx%d of a %dx%d picture", gridWidth, gridHeight, width, height)
	}
	compare("normalized latent", rows, interleaved(read("normalized"), latent*4))
}

// TestMOSSEncoderCompact holds the native encoder to the pinned diffusers source
// on the compact checkpoint: the moments' mean and the normalized packed latent.
func TestMOSSEncoderCompact(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	encoderGoldenCheck(t, root, "encoder-compact.safetensors", "6ff0deefa4a0b0b0a3f3ccab19e79d008bff070ae26a79801af0535c559def3f", 32, 48, 1)
}
