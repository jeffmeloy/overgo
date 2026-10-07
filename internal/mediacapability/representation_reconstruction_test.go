//go:build modeltest

package mediacapability

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/capabilityruntime"
	"overgo/internal/dataroot"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// TestPictureRoundTripsThroughTheRepresentationAutoencoder serves the RAE
// collection through the reconstruction capability: the collection resolves
// to its decoder, its latent statistics and the encoder beside it, on the
// host topology; a picture comes back as a PNG of the pair's own size that is
// far closer to what went in than one flat colour is; and the same request
// answers with the same picture.
func TestPictureRoundTripsThroughTheRepresentationAutoencoder(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	directory := roots.ResolveModelPath("RAE-collections")
	collection, held, err := rae.ResolveCollection(directory)
	if err != nil || !held {
		testskip.NotApplicable(t, fmt.Sprintf("no RAE collection with its encoder is at %s: %v", directory, err))
	}
	if _, err := os.Stat(filepath.Join(collection.EncoderDirectory(), "model.safetensors")); err != nil {
		testskip.NotApplicable(t, "the collection's encoder weights are absent: "+collection.EncoderDirectory())
	}
	testskip.Short(t, "loads the collection's 1.7 GB decoder")
	capability := Catalog[recipe.TaskReconstruction]
	source, err := capability.Resolve(directory)
	if err != nil {
		t.Fatal(err)
	}
	if files := len(source.Inventory.Manifest.Components); files != 5 || source.Define == nil {
		t.Fatalf("the pair resolves to %d files, want its decoder, statistics and the encoder's three", files)
	}
	definition, _, err := source.Define(source.Inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	if stage := program.Stages()[0]; stage.Module.ID != modelrecipe.ModuleReconstructRepresentation || definition.Nodes[0].Placement != recipe.PlacementHost {
		t.Fatalf("the pair compiles to entry module %s placed on %s", stage.Module.ID, definition.Nodes[0].Placement)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// Two slow gradients under a bright disc, at the size the pair works at.
	const side = 256
	picture := image.NewRGBA(image.Rect(0, 0, side, side))
	for y := range side {
		for x := range side {
			pixel := color.RGBA{R: uint8(40 + x/2), G: uint8(30 + y/2), B: uint8(200 - (x+y)/4), A: math.MaxUint8}
			if dx, dy := x-side/2, y-side/2; dx*dx+dy*dy < side*side/16 {
				pixel = color.RGBA{R: 240, G: 220, B: 90, A: math.MaxUint8}
			}
			picture.SetRGBA(x, y, pixel)
		}
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, picture); err != nil {
		t.Fatal(err)
	}
	request, err := json.Marshal(latentvideo.ReconstructionRequest{Image: encoded.Bytes()})
	if err != nil {
		t.Fatal(err)
	}
	reconstruct := func() latentimage.EncodedImage {
		t.Helper()
		output, err := capability.Execute(t.Context(), store, directory, modelrecipe.CapabilityEvidenceSelection{Program: program}, string(request))
		if err != nil {
			t.Fatal(err)
		}
		reconstruction, ok := capabilityruntime.Unwrap(output).(latentimage.EncodedImage)
		if !ok {
			t.Fatalf("reconstruction output is %T", capabilityruntime.Unwrap(output))
		}
		return reconstruction
	}
	first := reconstruct()
	decoded, err := png.Decode(bytes.NewReader(first.Data))
	if err != nil {
		t.Fatal(err)
	}
	if first.Height != side || first.Width != side || decoded.Bounds() != picture.Bounds() {
		t.Fatalf("the picture came back as %dx%d, PNG %v", first.Height, first.Width, decoded.Bounds())
	}
	var squared, flat float64
	var mean [3]float64
	for y := range side {
		for x := range side {
			pixel := picture.RGBAAt(x, y)
			for channel, value := range []uint8{pixel.R, pixel.G, pixel.B} {
				mean[channel] += float64(value) / (side * side)
			}
		}
	}
	for y := range side {
		for x := range side {
			want := picture.RGBAAt(x, y)
			gotRed, gotGreen, gotBlue, _ := decoded.At(x, y).RGBA()
			for channel, pair := range [][2]float64{{float64(want.R), float64(gotRed >> 8)}, {float64(want.G), float64(gotGreen >> 8)}, {float64(want.B), float64(gotBlue >> 8)}} {
				squared += (pair[0] - pair[1]) * (pair[0] - pair[1])
				flat += (pair[0] - mean[channel]) * (pair[0] - mean[channel])
			}
		}
	}
	values := float64(3 * side * side)
	ratio, control := 10*math.Log10(math.MaxUint8*math.MaxUint8*values/squared), 10*math.Log10(math.MaxUint8*math.MaxUint8*values/flat)
	t.Logf("a %dx%d picture round-trips at %.2f dB; one flat colour gives %.2f dB", side, side, ratio, control)
	if ratio < control+4 {
		t.Fatalf("the round trip gives %.2f dB, within 4 of the %.2f a flat colour gives", ratio, control)
	}
	if again := reconstruct(); !bytes.Equal(again.Data, first.Data) {
		t.Fatal("the same request answered with a different picture")
	}
}
