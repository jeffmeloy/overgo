package projector

import (
	"context"
	"image"
	"image/color"
	"testing"

	"overgo/internal/tensor/reference"
)

// parityRunners opens the same fixture on the CPU and CUDA backends;
// both close with the test.
func parityRunners[T Projector](t *testing.T, path string) (T, T) {
	t.Helper()
	cpu, err := openImageProjectorAs[T](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cpu.Close() })
	cuda, err := openImageProjectorAs[T](path, OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cuda.Close() })
	return cpu, cuda
}

// patternedRGBA fills a rectangle with a deterministic per-pixel color
// so parity inputs exercise every channel without random state.
func patternedRGBA(width, height int, pixel func(x, y int) color.RGBA) *image.RGBA {
	input := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			input.SetRGBA(x, y, pixel(x, y))
		}
	}
	return input
}

// referenceParityCase runs one reference-output projector's CUDA gate:
// the same fixture encodes the same patterned image on both backends
// and the embeddings must agree within tolerance.
func referenceParityCase[T interface {
	Projector
	EncodeImage(context.Context, image.Image) (reference.Value, error)
}](t *testing.T, name, path string, input image.Image, tolerance float32) {
	t.Helper()
	cpu, cuda := parityRunners[T](t, path)
	want, err := cpu.EncodeImage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cuda.EncodeImage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, name, got.Data, want.Data, tolerance)
}
