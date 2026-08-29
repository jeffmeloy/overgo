package trainingdata

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"slices"
	"testing"

	"overgo/internal/recipecontract"
)

func TestImageProcessorEmitsNormalizedCHW(t *testing.T) {
	imageData := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	imageData.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 255})
	imageData.SetNRGBA(1, 0, color.NRGBA{G: 255, B: 255, A: 255})
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, imageData); err != nil {
		t.Fatal(err)
	}
	example, err := ImageProcessor(RoleTarget)(t.Context(), RawRecord{ID: "image/0", Group: "image/0", Data: encoded.Bytes()})
	if err != nil {
		t.Fatal(err)
	}
	if len(example.Values) != 1 || example.Values[0].Modality != recipecontract.ModalityImage ||
		example.Values[0].Role != RoleTarget || !slices.Equal(example.Values[0].Shape, []int{3, 1, 2}) {
		t.Fatalf("example = %+v", example)
	}
	values, err := Float32(example.Values[0])
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, -1, -1, 1, -1, 1}
	for index := range want {
		if values[index] != want[index] {
			t.Fatalf("value[%d] = %g, want %g", index, values[index], want[index])
		}
	}
}
