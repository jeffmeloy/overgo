package rae

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// generationReference is fixtures/rae/generation.json: seeded noise, a class
// and a grid, and the image the reference sampled and decoded from them.
type generationReference struct {
	Source     string          `json:"source"`
	Commit     string          `json:"reference_commit"`
	Generator  string          `json:"generator_sha256"`
	Checkpoint string          `json:"checkpoint"`
	Decoder    string          `json:"decoder"`
	Statistics string          `json:"statistics"`
	Seed       int             `json:"seed"`
	Label      int             `json:"label"`
	Steps      int             `json:"steps"`
	Noise      referenceTensor `json:"noise"`
	Image      referenceTensor `json:"image"`
}

// TestRepresentationAutoencoderGeneration samples the reference's class from
// its seeded noise with the DINOv2-B ImageNet-256 DiTDH-S and decodes the
// latent with the ViT-XL decoder, and holds every pixel to the image the
// reference sampled and decoded from the same noise.
func TestRepresentationAutoencoderGeneration(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	fixtures := filepath.Join(root, "fixtures", "rae")
	var reference generationReference
	if err := jsonfile.DecodeStrict(filepath.Join(fixtures, "generation.json"), &reference); err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	collection := roots.ResolveModelPath("RAE-collections")
	encoder := roots.ResolveModelPath("dinov2-with-registers-base")
	checkpoint := filepath.Join(collection, filepath.FromSlash(reference.Checkpoint))
	decoderPath := filepath.Join(collection, filepath.FromSlash(reference.Decoder))
	for _, required := range []string{checkpoint, decoderPath, filepath.Join(encoder, "preprocessor_config.json")} {
		if _, err := os.Stat(required); err != nil {
			testskip.NotApplicable(t, "the RAE collection or its encoder is absent: "+required)
		}
	}
	testskip.Short(t, "loads the 1.7 GB ViT-XL decoder")
	model, err := loadDiT(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := loadDecoder(decoderPath, filepath.Join(collection, filepath.FromSlash(reference.Statistics)), encoder)
	if err != nil {
		t.Fatal(err)
	}
	got, err := generate(model, decoder, loadReference(t, fixtures, reference.Noise), reference.Label, reference.Steps)
	if err != nil {
		t.Fatal(err)
	}
	worst, _ := compareReference(t, got, loadReference(t, fixtures, reference.Image))
	t.Logf("max |generated - reference| = %.3g over %d pixels", worst, len(got))
	if worst > 1e-3 {
		t.Fatalf("generated image differs from the reference by up to %.3g", worst)
	}
}
