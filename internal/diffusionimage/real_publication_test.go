//go:build modeltest

package diffusionimage

import (
	"bytes"
	"os"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestRealCheckpointPublishesReferencePNG(t *testing.T) {
	model := loadArtifactModel(t)
	plan := testcheck.Must(model.prepare(t.Context(), Request{Seed: 7, Steps: 2, Height: 64, Width: 64}))(t)
	features := testcheck.Must(model.integrate(t.Context(), plan))(t)
	encoded := testcheck.Must(model.decode(t.Context(), features))(t)
	referencePath := testutil.FixturePath(t, "simplediffusion", "uvit_direct_seed7_steps2_64.png")
	reference, err := os.ReadFile(referencePath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: reference image %s: %v", referencePath, err)
	}
	if encoded.MediaType != "image/png" || encoded.Channels != 3 || encoded.Height != 64 || encoded.Width != 64 || !bytes.Equal(encoded.Data, reference) {
		t.Fatalf("SimpleDiffusion PNG differs: media=%s shape=%dx%dx%d bytes=%d want=%d", encoded.MediaType, encoded.Channels, encoded.Height, encoded.Width, len(encoded.Data), len(reference))
	}
	t.Logf("real SimpleDiffusion PNG: seed=7 steps=2 size=64x64 encoded_bytes=%d exact_reference=true range=[%.6f,%.6f]", len(encoded.Data), encoded.Minimum, encoded.Maximum)
}
