//go:build modeltest

package rae

import (
	"os"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"path/filepath"
	"testing"
)

// TestRAEDecoderMatchesReference decodes the reference's seeded latent with
// the DINOv2-with-registers ViT-XL decoder and its ImageNet latent
// statistics, and holds every pixel to the image the reference decoder made
// of the same latent.
func TestRAEDecoderMatchesReference(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	fixtures := filepath.Join(root, "fixtures", "rae")
	var reference decoderReference
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(fixtures, "decoder.json"), &reference))
	roots, err := dataroot.Resolve(root)
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	got, err := decoder.decode(loadReference(t, fixtures, reference.Latent))
	testcheck.NoError(t, err)
	worst, _ := compareReference(t, got, loadReference(t, fixtures, reference.Image))
	t.Logf("max |decoded - reference| = %.3g over %d pixels", worst, len(got))
	if worst > 1e-3 {
		t.Fatalf("decoded image differs from the reference by up to %.3g", worst)
	}
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
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(fixtures, "generation.json"), &reference))
	roots, err := dataroot.Resolve(root)
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	decoder, err := loadDecoder(decoderPath, filepath.Join(collection, filepath.FromSlash(reference.Statistics)), encoder)
	testcheck.NoError(t, err)
	latent, err := model.sample(t.Context(), loadReference(t, fixtures, reference.Noise), reference.Label, reference.Steps, 1)
	testcheck.NoError(t, err)
	got, err := decoder.decode(latent)
	testcheck.NoError(t, err)
	worst, _ := compareReference(t, got, loadReference(t, fixtures, reference.Image))
	t.Logf("max |generated - reference| = %.3g over %d pixels", worst, len(got))
	if worst > 1e-3 {
		t.Fatalf("generated image differs from the reference by up to %.3g", worst)
	}
}
