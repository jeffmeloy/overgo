//go:build modeltest

package mediacapability

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image/png"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/capabilityruntime"
	"overgo/internal/dataroot"
	"overgo/internal/fixtureasset"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/rae"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// raeGeneration is what a served RAE generation is held to in
// fixtures/rae/generation.json: the DiT checkpoint the reference sampled
// with, the class, seed and grid points it sampled, and the image it
// decoded, channel-first in [0, 1].
type raeGeneration struct {
	Checkpoint string `json:"checkpoint"`
	Seed       int64  `json:"seed"`
	Label      int    `json:"label"`
	Steps      int    `json:"steps"`
	Image      struct {
		File   string `json:"file"`
		Shape  []int  `json:"shape"`
		SHA256 string `json:"sha256"`
	} `json:"image"`
}

// TestRAEImageCapabilityGenerates serves the RAE collection through the
// image-gen capability: the collection resolves to its reviewed sampler's
// DiT, decoder and latent statistics on the RAE topology, a request naming
// the generation golden's class, seed and grid points returns the PNG of the
// image the reference sampled and decoded from that seed, and a class the
// DiT does not know is refused.
func TestRAEImageCapabilityGenerates(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	fixtures := filepath.Join(root, "fixtures", "rae")
	var reference raeGeneration
	testcheck.NoError(t, jsonfile.Decode(filepath.Join(fixtures, "generation.json"), &reference))
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	directory := roots.ResolveModelPath("RAE-collections")
	collection, held, err := rae.ResolveCollection(directory)
	if err != nil || !held {
		testskip.NotApplicable(t, fmt.Sprintf("no RAE collection with its encoder is at %s: %v", directory, err))
	}
	if want := filepath.Join(filepath.Clean(directory), filepath.FromSlash(reference.Checkpoint)); collection.Checkpoint != want {
		t.Fatalf("the served DiT %s is not the one the generation golden sampled with, %s", collection.Checkpoint, want)
	}
	testskip.Short(t, "samples with the collection's DiT and decodes with its 1.7 GB decoder")
	store := overgodbtest.Open(t, t.TempDir())
	capability := Catalog[recipe.TaskImageGen]
	source := testcheck.Must(capability.Resolve(t.Context(), directory))(t)
	definition, _, err := source.Define(source.Inventory.Manifest.ID)
	testcheck.NoError(t, err)
	program := testcheck.Must(modelrecipe.CompileCapability(definition))(t)
	if entry := program.Stages()[0].Module.ID; entry != modelrecipe.ModuleRAEImagePrepare {
		t.Fatalf("the collection compiles to entry module %s", entry)
	}
	generate := func(request rae.Request) (latentimage.EncodedImage, error) {
		raw := testcheck.Must(json.Marshal(request))(t)
		output, err := capability.Execute(t.Context(), store, directory, modelrecipe.CapabilityEvidenceSelection{Program: program}, string(raw))
		if err != nil {
			return latentimage.EncodedImage{}, err
		}
		image, ok := capabilityruntime.Unwrap(output).(latentimage.EncodedImage)
		if !ok {
			t.Fatalf("image-gen output is %T", capabilityruntime.Unwrap(output))
		}
		return image, nil
	}
	image := testcheck.Must(generate(rae.Request{Class: reference.Label, Seed: reference.Seed, Steps: reference.Steps}))(t)
	channels, height, width := reference.Image.Shape[0], reference.Image.Shape[1], reference.Image.Shape[2]
	golden := testcheck.Must(fixtureasset.LoadF32(fixtures, reference.Image.File, reference.Image.SHA256, channels*height*width))(t)
	decoded := testcheck.Must(png.Decode(bytes.NewReader(image.Data)))(t)
	if bounds := decoded.Bounds(); bounds.Dx() != width || bounds.Dy() != height || image.Channels != channels {
		t.Fatalf("generated a %dx%d image of %d channels, the reference is %dx%d of %d", bounds.Dx(), bounds.Dy(), image.Channels, width, height, channels)
	}
	// Each byte is the reference's pixel quantized as the image encoding
	// quantizes its own: [0, 1] mapped to [-1, 1], then to 0..255.
	worst, differing := 0, 0
	for y := range height {
		for x := range width {
			red, green, blue, _ := decoded.At(x, y).RGBA()
			for channel, value := range []uint32{red >> 8, green >> 8, blue >> 8} {
				pixel := float64(2*golden[(channel*height+y)*width+x]-1) + 1
				want := int(min(max(pixel*255/2, 0), 255))
				if delta := max(int(value)-want, want-int(value)); delta > 0 {
					worst, differing = max(worst, delta), differing+1
				}
			}
		}
	}
	t.Logf("class %d from seed %d over %d grid points: %d of %d bytes differ from the reference image, by at most %d",
		reference.Label, reference.Seed, reference.Steps, differing, channels*height*width, worst)
	if worst > 1 {
		t.Fatalf("the generated image differs from the reference by up to %d levels", worst)
	}
	if _, err := generate(rae.Request{Class: 1000, Seed: reference.Seed, Steps: reference.Steps}); err == nil || !strings.Contains(err.Error(), "outside the DiT's") {
		t.Fatalf("a class the DiT does not know was sampled: %v", err)
	}
}
