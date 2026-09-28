package rae

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/fixtureasset"
	"overgo/internal/jsonfile"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// referenceTensor is one golden tensor the fixture manifest pins.
type referenceTensor struct {
	File   string `json:"file"`
	Shape  []int  `json:"shape"`
	SHA256 string `json:"sha256"`
}

// decoderReference is fixtures/rae/decoder.json: a seeded normalized latent
// and the image the reference decoder made of it.
type decoderReference struct {
	Source     string          `json:"source"`
	Commit     string          `json:"reference_commit"`
	Generator  string          `json:"generator_sha256"`
	Encoder    string          `json:"encoder"`
	Decoder    string          `json:"decoder"`
	Statistics string          `json:"statistics"`
	Seed       int             `json:"seed"`
	Latent     referenceTensor `json:"latent"`
	Image      referenceTensor `json:"image"`
}

func loadReference(t *testing.T, directory string, tensor referenceTensor) []float32 {
	t.Helper()
	elements := 1
	for _, extent := range tensor.Shape {
		elements *= extent
	}
	values, err := fixtureasset.LoadF32(directory, tensor.File, tensor.SHA256, elements)
	if err != nil {
		t.Fatal(err)
	}
	return values
}

// TestRAEDecoderMatchesReference decodes the reference's seeded latent with
// the DINOv2-with-registers ViT-XL decoder and its ImageNet latent
// statistics, and holds every pixel to the image the reference decoder made
// of the same latent.
func TestRAEDecoderMatchesReference(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	fixtures := filepath.Join(root, "fixtures", "rae")
	var reference decoderReference
	if err := jsonfile.DecodeStrict(filepath.Join(fixtures, "decoder.json"), &reference); err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	collection := roots.ResolveModelPath("RAE-collections")
	encoder := roots.ResolveModelPath("dinov2-with-registers-base")
	for _, required := range []string{filepath.Join(collection, filepath.FromSlash(reference.Decoder)), filepath.Join(encoder, "preprocessor_config.json")} {
		if _, err := os.Stat(required); err != nil {
			testskip.NotApplicable(t, "the RAE collection or its encoder is absent: "+required)
		}
	}
	testskip.Short(t, "loads the 1.7 GB ViT-XL decoder")
	decoder, err := loadDecoder(filepath.Join(collection, filepath.FromSlash(reference.Decoder)),
		filepath.Join(collection, filepath.FromSlash(reference.Statistics)), encoder)
	if err != nil {
		t.Fatal(err)
	}
	latent := loadReference(t, fixtures, reference.Latent)
	want := loadReference(t, fixtures, reference.Image)
	got, err := decoder.decode(latent)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("decoded %d pixels, reference has %d", len(got), len(want))
	}
	var worst float64
	for index := range got {
		worst = max(worst, math.Abs(float64(got[index]-want[index])))
	}
	t.Logf("max |decoded - reference| = %.3g over %d pixels", worst, len(got))
	if worst > 1e-3 {
		t.Fatalf("decoded image differs from the reference by up to %.3g", worst)
	}
}
