//go:build modeltest

package rae

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// roundTripMarginDB is how far the pair's round trip of the test picture must
// beat a picture of one flat colour, the best answer that reads nothing of
// what went in. A representation autoencoder keeps what a picture shows
// rather than its pixels, so its ratio is modest, about 19 dB here; tokens
// read in the wrong order or a picture normalized twice do no better than
// the flat colour.
const roundTripMarginDB = 4

// TestRepresentationAutoencoderRoundTripsAPicture holds the two halves, each
// held to its reference on its own, to working together: a picture goes
// through the with-registers encoder, whose tokens the ViT-XL decoder turns
// back into a picture close to the one that went in; and a picture of
// another size is scaled and cropped to the size the pair works at.
func TestRepresentationAutoencoderRoundTripsAPicture(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	collection := roots.ResolveModelPath("RAE-collections")
	resolved, held, err := ResolveCollection(collection)
	if err != nil || !held {
		testskip.NotApplicable(t, "the RAE collection or its encoder is absent: "+collection)
	}
	if _, err := os.Stat(filepath.Join(resolved.EncoderDirectory(), "model.safetensors")); err != nil {
		testskip.NotApplicable(t, "the collection's encoder weights are absent: "+resolved.EncoderDirectory())
	}
	testskip.Short(t, "loads the 1.7 GB ViT-XL decoder")
	pair, err := LoadAutoencoder(collection)
	if err != nil {
		t.Fatal(err)
	}
	side := pair.Side()
	if channels, grid := pair.Latent(); side != 256 || channels != 768 || grid != 16 {
		t.Fatalf("the pair works at %d with a latent of %d channels over a %d grid", side, channels, grid)
	}
	// Two slow gradients under a bright disc, channel-first in [0, 1].
	plane := side * side
	picture := make([]float32, 3*plane)
	shade := func(x, y int) [3]float32 {
		if dx, dy := x-side/2, y-side/2; dx*dx+dy*dy < side*side/16 {
			return [3]float32{0.94, 0.86, 0.35}
		}
		return [3]float32{0.15 + 0.6*float32(x)/float32(side), 0.1 + 0.7*float32(y)/float32(side), 0.8 - 0.3*float32(x+y)/float32(side)}
	}
	for y := range side {
		for x := range side {
			for channel, value := range shade(x, y) {
				picture[channel*plane+y*side+x] = value
			}
		}
	}
	back, err := pair.RoundTrip(t.Context(), picture)
	if err != nil {
		t.Fatal(err)
	}
	if len(back) != len(picture) {
		t.Fatalf("%d values came back for %d", len(back), len(picture))
	}
	var squared, flat float64
	for channel := range 3 {
		var mean float64
		for _, value := range picture[channel*plane:][:plane] {
			mean += float64(value) / float64(plane)
		}
		for index, value := range picture[channel*plane:][:plane] {
			difference := float64(value) - float64(min(max(back[channel*plane+index], 0), 1))
			squared += difference * difference
			flat += (float64(value) - mean) * (float64(value) - mean)
		}
	}
	ratio, control := 10*math.Log10(float64(len(picture))/squared), 10*math.Log10(float64(len(picture))/flat)
	t.Logf("a %dx%d picture round-trips at %.2f dB; one flat colour gives %.2f dB", side, side, ratio, control)
	if ratio < control+roundTripMarginDB {
		t.Fatalf("the round trip gives %.2f dB, within %d of the %.2f a flat colour gives", ratio, roundTripMarginDB, control)
	}
	// A wide picture is scaled to the pair's height and its middle taken.
	wide := image.NewRGBA(image.Rect(0, 0, 3*side/2, side/2))
	for y := range wide.Bounds().Dy() {
		for x := range wide.Bounds().Dx() {
			wide.SetRGBA(x, y, color.RGBA{R: uint8(x % 256), G: uint8(2 * y % 256), B: 90, A: math.MaxUint8})
		}
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, wide); err != nil {
		t.Fatal(err)
	}
	reconstruction, err := pair.Reconstruct(t.Context(), encoded.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if reconstruction.Height != side || reconstruction.Width != side {
		t.Fatalf("a wide picture came back as %dx%d, want the pair's %dx%d", reconstruction.Height, reconstruction.Width, side, side)
	}
	if _, err := pair.Reconstruct(t.Context(), []byte("no picture")); err == nil {
		t.Fatal("bytes that are no picture were reconstructed")
	}
}
