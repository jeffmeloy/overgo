package vitencoder

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/modeltest"
)

// golden is what the reference produced for the synthetic image: the resized
// bytes' digest, a sample of the normalized pixels, the class token and two
// patch tokens after the final norm (testdata/dinov2_golden.json, written by
// the model repository's own torch code with its image processor's PIL
// resize).
type golden struct {
	Resized struct {
		Height int    `json:"height"`
		Width  int    `json:"width"`
		SHA256 string `json:"sha256"`
	} `json:"resized"`
	PixelIndices []int     `json:"pixel_indices"`
	Pixels       []float32 `json:"pixels"`
	Class        []float32 `json:"cls"`
	FirstPatch   []float32 `json:"first_patch"`
	LastPatch    []float32 `json:"last_patch"`
}

// syntheticImage is the golden's input, generated the same way on both sides
// so no image file is needed: (x*7 + y*13 + c*29 + (x*y)%17) % 256.
func syntheticImage(height, width int) []uint8 {
	pixels := make([]uint8, height*width*channels)
	for y := range height {
		for x := range width {
			for c := range channels {
				pixels[(y*width+x)*channels+c] = uint8((x*7 + y*13 + c*29 + (x*y)%17) % 256)
			}
		}
	}
	return pixels
}

// TestDinov2MatchesReference holds the encoder and its image processor to the
// reference: the resized image byte for byte, the normalized pixels exactly,
// and the class and patch tokens to float32 accumulation-order tolerance.
func TestDinov2MatchesReference(t *testing.T) {
	t.Parallel()
	var reference golden
	if err := jsonfile.Decode(filepath.Join("testdata", "dinov2_golden.json"), &reference); err != nil {
		t.Fatal(err)
	}
	image := syntheticImage(256, 320)
	resized, err := resizeBicubic(image, 256, 320, reference.Resized.Height, reference.Resized.Width)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(resized)
	if hex.EncodeToString(digest[:]) != reference.Resized.SHA256 {
		t.Fatal("the bicubic resize differs from PIL's")
	}
	encoder, err := Load(modeltest.Directory(t, "webssl-dino300m-full2b-224"))
	if err != nil {
		t.Fatal(err)
	}
	pixels, err := encoder.preprocess.pixels(image, 256, 320)
	if err != nil {
		t.Fatal(err)
	}
	for index, at := range reference.PixelIndices {
		if pixels[at] != reference.Pixels[index] {
			t.Fatalf("normalized pixel %d = %v, reference %v", at, pixels[at], reference.Pixels[index])
		}
	}
	tokens, err := encoder.encode(pixels)
	if err != nil {
		t.Fatal(err)
	}
	hidden := encoder.geometry.Hidden
	last := len(tokens)/hidden - 1
	for _, check := range []struct {
		name string
		got  []float32
		want []float32
	}{
		{"class token", tokens[:hidden], reference.Class},
		{"first patch", tokens[hidden : 2*hidden], reference.FirstPatch},
		{"last patch", tokens[last*hidden:], reference.LastPatch},
	} {
		worst, cosine := compare(check.got, check.want)
		t.Logf("%s: max abs diff %.3g, cosine %.9f", check.name, worst, cosine)
		if worst > 2e-3 || cosine < 0.999999 {
			t.Errorf("%s differs from the reference: max abs diff %.3g, cosine %.9f", check.name, worst, cosine)
		}
	}
}

func compare(got, want []float32) (float64, float64) {
	var worst, dot, gotNorm, wantNorm float64
	for index := range want {
		g, w := float64(got[index]), float64(want[index])
		worst = max(worst, math.Abs(g-w))
		dot += g * w
		gotNorm += g * g
		wantNorm += w * w
	}
	return worst, dot / math.Sqrt(gotNorm*wantNorm)
}
