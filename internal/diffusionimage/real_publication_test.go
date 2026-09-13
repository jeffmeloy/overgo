//go:build modeltest

package diffusionimage

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestRealCheckpointPublishesReferencePNG(t *testing.T) {
	model := loadArtifactModel(t)
	plan, err := model.prepare(t.Context(), Request{Seed: 7, Steps: 2, Height: 64, Width: 64})
	if err != nil {
		t.Fatal(err)
	}
	features, err := model.integrate(t.Context(), plan)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := model.decode(t.Context(), features)
	if err != nil {
		t.Fatal(err)
	}
	referencePath := filepath.Join(filepath.Dir(filepath.Dir(artifactDir(t))), "docs", "image_gen_samples", "uvit_direct_seed7_steps2_64.png")
	reference, err := os.ReadFile(referencePath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: reference image %s: %v", referencePath, err)
	}
	if encoded.MediaType != "image/png" || encoded.Channels != 3 || encoded.Height != 64 || encoded.Width != 64 || !bytes.Equal(encoded.Data, reference) {
		t.Fatalf("SimpleDiffusion PNG differs: media=%s shape=%dx%dx%d bytes=%d want=%d", encoded.MediaType, encoded.Channels, encoded.Height, encoded.Width, len(encoded.Data), len(reference))
	}
	t.Logf("real SimpleDiffusion PNG: seed=7 steps=2 size=64x64 encoded_bytes=%d exact_reference=true range=[%.6f,%.6f]", len(encoded.Data), encoded.Minimum, encoded.Maximum)
}
