package projector

import (
	"context"
	"image"
	"image/color"
	"testing"

	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// parityRunners opens the same fixture on the CPU and CUDA backends;
// both close with the test.
func parityRunners[T Projector](t *testing.T, path string, options OpenOptions) (T, T) {
	t.Helper()
	options.CUDA = false
	cpu := testcheck.Must(openImageProjectorAs[T](path, options))(t)
	t.Cleanup(func() { _ = cpu.Close() })
	options.CUDA = true
	cuda := testcheck.Must(openImageProjectorAs[T](path, options))(t)
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
	want := testcheck.Must(cpu.EncodeImage(t.Context(), input))(t)
	got := testcheck.Must(cuda.EncodeImage(t.Context(), input))(t)
	compareFloat32Tolerance(t, name, got.Data, want.Data, tolerance)
}

// qwenImageVideoParityCase keeps the pinned 4x4 image and four-frame protocol.
func qwenImageVideoParityCase[T interface {
	Projector
	EncodeImage(context.Context, image.Image, mediaPixelBudget) (qwen3VLOutput, error)
	EncodeFrames(context.Context, []image.Image, mediaPixelBudget) (qwen3VLOutput, error)
}](t *testing.T, name, path string) {
	t.Helper()
	cpu, cuda := parityRunners[T](t, path, fixtureMediaPreprocessOptions(t, OpenOptions{}))
	input := patternedRGBA(4, 4, func(x, y int) color.RGBA {
		return color.RGBA{R: uint8(x * 40), G: uint8(y * 40), B: 80, A: fixtureOpaqueAlpha}
	})
	options := mediaPixelBudget{MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget}
	wantImage := testcheck.Must(cpu.EncodeImage(t.Context(), input, options))(t)
	gotImage := testcheck.Must(cuda.EncodeImage(t.Context(), input, options))(t)
	compareFloat32Tolerance(t, name+" image", gotImage.Embeddings.Data, wantImage.Embeddings.Data, 2e-3)
	frames := []image.Image{input, input, input, input}
	wantVideo := testcheck.Must(cpu.EncodeFrames(t.Context(), frames, options))(t)
	gotVideo := testcheck.Must(cuda.EncodeFrames(t.Context(), frames, options))(t)
	compareFloat32Tolerance(t, name+" video", gotVideo.Embeddings.Data, wantVideo.Embeddings.Data, 2e-3)
}
