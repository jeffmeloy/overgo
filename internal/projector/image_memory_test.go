package projector

import (
	"context"
	"errors"
	"image"
	"image/color"
	"testing"
)

// TestRememberImagesEncodesAPictureOnce holds a session's image encoder to encoding a picture once while it is among
// the last pictures it was given: the same pixels in another image value are found, another picture is encoded, the
// oldest of more than it keeps is encoded again, and a failed encoding is not kept.
func TestRememberImagesEncodesAPictureOnce(t *testing.T) {
	t.Parallel()
	encoded := 0
	fail := false
	encode := rememberImages(func(_ context.Context, source image.Image) (imagePromptItem, error) {
		if fail {
			return imagePromptItem{}, errors.New("encoding failed")
		}
		encoded++
		return imagePromptItem{Count: source.Bounds().Dx()}, nil
	})
	picture := func(width int, shade uint8) image.Image {
		made := image.NewNRGBA(image.Rect(0, 0, width, width))
		made.SetNRGBA(0, 0, color.NRGBA{R: shade, A: 255})
		return made
	}
	gray := func(shade uint8) image.Image {
		made := image.NewGray(image.Rect(0, 0, 3, 3))
		made.SetGray(1, 1, color.Gray{Y: shade})
		return made
	}
	ask := func(source image.Image, want int) {
		t.Helper()
		item, err := encode(t.Context(), source)
		if err != nil || item.Count != source.Bounds().Dx() || encoded != want {
			t.Fatalf("encodings = %d, item %+v, %v; want %d", encoded, item, err, want)
		}
	}
	ask(picture(2, 1), 1)
	ask(picture(2, 1), 1)
	ask(picture(2, 2), 2)
	ask(gray(7), 3)
	ask(gray(7), 3)
	ask(gray(8), 4)
	ask(picture(5, 1), 5)
	ask(picture(2, 2), 5)
	ask(picture(2, 1), 6)
	fail = true
	if _, err := encode(t.Context(), picture(9, 9)); err == nil {
		t.Fatal("a failed encoding gave no error")
	}
	fail = false
	ask(picture(9, 9), 7)
}
