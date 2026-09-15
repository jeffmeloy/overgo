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
func parityRunners[T Projector](t *testing.T, path string, options OpenOptions) (T, T) {
	t.Helper()
	options.CUDA = false
	cpu, err := openImageProjectorAs[T](path, options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cpu.Close() })
	options.CUDA = true
	cuda, err := openImageProjectorAs[T](path, options)
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
	for y := range height {
		for x := range width {
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
	cpu, cuda := parityRunners[T](t, path, OpenOptions{})
	want, err := cpu.EncodeImage(t.Context(), input)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cuda.EncodeImage(t.Context(), input)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, name, got.Data, want.Data, tolerance)
}

// qwenImageVideoParityCase keeps the pinned 4x4 image and four-frame protocol.
func qwenImageVideoParityCase[T interface {
	Projector
	EncodeImage(context.Context, image.Image, MediaPixelBudget) (Qwen3VLOutput, error)
	EncodeFrames(context.Context, []image.Image, MediaPixelBudget) (Qwen3VLOutput, error)
}](t *testing.T, name, path string) {
	t.Helper()
	cpu, cuda := parityRunners[T](t, path, fixtureMediaPreprocessOptions(t, OpenOptions{}))
	input := patternedRGBA(4, 4, func(x, y int) color.RGBA {
		return color.RGBA{R: uint8(x * 40), G: uint8(y * 40), B: 80, A: fixtureOpaqueAlpha}
	})
	options := MediaPixelBudget{MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget}
	wantImage, err := cpu.EncodeImage(t.Context(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	gotImage, err := cuda.EncodeImage(t.Context(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, name+" image", gotImage.Embeddings.Data, wantImage.Embeddings.Data, 2e-3)
	frames := []image.Image{input, input, input, input}
	wantVideo, err := cpu.EncodeFrames(t.Context(), frames, options)
	if err != nil {
		t.Fatal(err)
	}
	gotVideo, err := cuda.EncodeFrames(t.Context(), frames, options)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, name+" video", gotVideo.Embeddings.Data, wantVideo.Embeddings.Data, 2e-3)
}
