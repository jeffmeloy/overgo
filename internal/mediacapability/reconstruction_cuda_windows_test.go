//go:build windows

package mediacapability

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"

	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// reconstructionFloorDB is the least peak signal-to-noise ratio the round
// trip of a test picture may give. The installed autoencoder gives well over
// it; a decode from the wrong latent statistics, a swapped channel or a
// shifted crop gives under 20.
const reconstructionFloorDB = 30

// TestPictureRoundTripsThroughAnInstalledAutoencoder runs the catalog's
// reconstruction end to end on the installed Wan autoencoder: the directory
// resolves to the autoencoder's one file, the recipe compiles, a picture
// whose sides are not multiples of the stride comes back as a PNG of its own
// size that is close to what went in, the same request answers with the same
// picture, and the resident autoencoder then takes a picture of another size.
func TestPictureRoundTripsThroughAnInstalledAutoencoder(t *testing.T) {
	cudatest.Require(t)
	directory := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
	capability := Catalog[recipe.TaskReconstruction]
	source, err := capability.Resolve(directory)
	if err != nil {
		t.Fatal(err)
	}
	modelID := source.Inventory.Manifest.ID
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskReconstruction, modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishArtifactExtent(t, store, modelID, latentvideo.AutoencoderWeights(directory))
	execution := modelrecipetest.CandidateExecution(t, store, program)
	reconstruct := func(picture *image.RGBA) []byte {
		t.Helper()
		var encoded bytes.Buffer
		if err := png.Encode(&encoded, picture); err != nil {
			t.Fatal(err)
		}
		request, err := json.Marshal(latentvideo.ReconstructionRequest{Image: encoded.Bytes()})
		if err != nil {
			t.Fatal(err)
		}
		output, err := capability.Execute(t.Context(), store, directory, execution, string(request))
		if err != nil {
			t.Fatal(err)
		}
		reconstruction, ok := capabilityruntime.Unwrap(output).(latentimage.EncodedImage)
		if !ok {
			t.Fatalf("reconstruction output type=%T", output)
		}
		height, width := picture.Bounds().Dy(), picture.Bounds().Dx()
		decoded, err := png.Decode(bytes.NewReader(reconstruction.Data))
		if err != nil {
			t.Fatal(err)
		}
		if reconstruction.Height != height || reconstruction.Width != width || decoded.Bounds() != picture.Bounds() {
			t.Fatalf("a %dx%d picture came back as %dx%d, PNG %v", height, width, reconstruction.Height, reconstruction.Width, decoded.Bounds())
		}
		var squared float64
		for y := range height {
			for x := range width {
				wantRed, wantGreen, wantBlue, _ := picture.At(x, y).RGBA()
				gotRed, gotGreen, gotBlue, _ := decoded.At(x, y).RGBA()
				for _, pair := range [][2]uint32{{wantRed, gotRed}, {wantGreen, gotGreen}, {wantBlue, gotBlue}} {
					difference := float64(pair[0]>>8) - float64(pair[1]>>8)
					squared += difference * difference
				}
			}
		}
		ratio := 10 * math.Log10(math.MaxUint8*math.MaxUint8/(squared/float64(3*height*width)))
		t.Logf("a %dx%d picture round-trips at %.2f dB", height, width, ratio)
		if ratio < reconstructionFloorDB {
			t.Fatalf("the round trip of a %dx%d picture gives %.2f dB, under the floor of %d", height, width, ratio, reconstructionFloorDB)
		}
		return reconstruction.Data
	}
	// 76 by 100 is extended to 80 by 104 for a stride of eight and cropped
	// back; 64 by 96 needs no extension.
	odd, even := testPicture(76, 100), testPicture(64, 96)
	first := reconstruct(odd)
	if again := reconstruct(odd); !bytes.Equal(again, first) {
		t.Fatal("the same request answered with a different picture")
	}
	reconstruct(even)
}

// testPicture draws two slow gradients under a bright disc: smooth ground
// and one hard edge, the two things a picture is made of.
func testPicture(height, width int) *image.RGBA {
	picture := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			pixel := color.RGBA{R: uint8(40 + 2*x), G: uint8(30 + 2*y), B: uint8(200 - x - y), A: math.MaxUint8}
			if dx, dy := x-width/2, y-height/2; dx*dx+dy*dy < height*height/9 {
				pixel = color.RGBA{R: 240, G: 220, B: 90, A: math.MaxUint8}
			}
			picture.SetRGBA(x, y, pixel)
		}
	}
	return picture
}
