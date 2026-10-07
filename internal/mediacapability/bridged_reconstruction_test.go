//go:build modeltest

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

// TestPictureRoundTripsThroughABridgedPair serves the installed bridged pair
// through the reconstruction capability: its directory resolves to the
// bridge, the encoder the bridge maps, that encoder's configuration and the
// decoder it feeds, on the host topology; a picture comes back as a PNG of
// the decoder's size; and an evaluation opens the same pair and is given the
// same picture back.
func TestPictureRoundTripsThroughABridgedPair(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	directory := roots.ResolveModelPath("webssl-dino-to-rae-bridge")
	if _, held, err := rae.ResolveBridgedPair(directory); err != nil || !held {
		testskip.NotApplicable(t, "no bridged pair with its encoder and collection is at "+directory)
	}
	testskip.Short(t, "loads the bridged encoder and the collection's 1.7 GB decoder")
	capability := Catalog[recipe.TaskReconstruction]
	source, err := capability.Resolve(directory)
	if err != nil {
		t.Fatal(err)
	}
	if files := len(source.Inventory.Manifest.Components); files != 7 || source.Define == nil {
		t.Fatalf("the bridged pair resolves to %d files, want its bridge, the encoder's three, the decoder's two and the decoder's image processor", files)
	}
	definition, _, err := source.Define(source.Inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	if entry := program.Stages()[0].Module.ID; entry != modelrecipe.ModuleReconstructRepresentation {
		t.Fatalf("the bridged pair compiles to entry module %s", entry)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
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
	output, err := capability.Execute(t.Context(), store, directory, modelrecipe.CapabilityEvidenceSelection{Program: program}, string(request))
	if err != nil {
		t.Fatal(err)
	}
	reconstruction, ok := capabilityruntime.Unwrap(output).(latentimage.EncodedImage)
	if !ok {
		t.Fatalf("reconstruction output is %T", capabilityruntime.Unwrap(output))
	}
	decoded, err := png.Decode(bytes.NewReader(reconstruction.Data))
	if err != nil {
		t.Fatal(err)
	}
	if reconstruction.Height != side || reconstruction.Width != side || decoded.Bounds() != picture.Bounds() {
		t.Fatalf("the picture came back as %dx%d, PNG %v", reconstruction.Height, reconstruction.Width, decoded.Bounds())
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
	// The same picture as an evaluation gives it: channel-first in [-1, 1].
	clip := latentvideo.SourceVideo{Pixels: make([]float32, 3*side*side), Channels: 3, Frames: 1, Height: side, Width: side}
	for y := range side {
		for x := range side {
			want := picture.RGBAAt(x, y)
			gotRed, gotGreen, gotBlue, _ := decoded.At(x, y).RGBA()
			for channel, pair := range [][2]float64{{float64(want.R), float64(gotRed >> 8)}, {float64(want.G), float64(gotGreen >> 8)}, {float64(want.B), float64(gotBlue >> 8)}} {
				squared += (pair[0] - pair[1]) * (pair[0] - pair[1])
				flat += (pair[0] - mean[channel]) * (pair[0] - mean[channel])
				clip.Pixels[(channel*side+y)*side+x] = float32(pair[0])/math.MaxUint8*2 - 1
			}
		}
	}
	values := float64(3 * side * side)
	ratio, control := 10*math.Log10(math.MaxUint8*math.MaxUint8*values/squared), 10*math.Log10(math.MaxUint8*math.MaxUint8*values/flat)
	// The bridge was fitted on frames of real clips, and this drawn picture is unlike
	// them: how close it comes back is logged, and judged by the held-out suite, not
	// here. What is held here is that the pair runs as the capability and as an
	// evaluation, and gives both the same picture.
	t.Logf("a drawn %dx%d picture round-trips at %.2f dB through the bridged pair; one flat colour gives %.2f dB", side, side, ratio, control)
	if math.IsNaN(ratio) || math.IsInf(ratio, 0) {
		t.Fatalf("the bridged round trip gives no finite ratio: %v", ratio)
	}
	opened, err := OpenAutoencoder(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close(t.Context())
	back, latent, err := opened.RoundTrip(t.Context(), clip)
	if err != nil {
		t.Fatal(err)
	}
	if latent.Channels != 768 || latent.Height != 16 || latent.Width != 16 || len(back.Pixels) != len(clip.Pixels) {
		t.Fatalf("the evaluation's round trip gave a latent of %+v and %d values", latent, len(back.Pixels))
	}
	var worst float64
	for y := range side {
		for x := range side {
			gotRed, gotGreen, gotBlue, _ := decoded.At(x, y).RGBA()
			for channel, shown := range []uint32{gotRed, gotGreen, gotBlue} {
				value := min(max((back.Pixels[(channel*side+y)*side+x]+1)/2, 0), 1) * math.MaxUint8
				worst = max(worst, math.Abs(float64(value)-float64(shown>>8)))
			}
		}
	}
	if worst > 1.5 {
		t.Fatalf("the evaluation's picture differs from the capability's by up to %.2f levels", worst)
	}
	if _, _, err := opened.RoundTrip(t.Context(), latentvideo.SourceVideo{Pixels: make([]float32, 3*8*8), Channels: 3, Frames: 1, Height: 8, Width: 8}); err == nil {
		t.Fatal("a picture of another size was taken")
	}
}
