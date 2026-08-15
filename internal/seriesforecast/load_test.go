package seriesforecast

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

// artifactDir resolves the real timesfm artifact through the data-root
// contract; absent artifact skips LOUDLY (UNAVAILABLE is never green — the
// skip names what was not tested).
func artifactDir(t *testing.T) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(roots.Models, "timesfm-2.5-200m-transformers")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("UNAVAILABLE: timesfm artifact absent at %s; dims-vs-artifact NOT verified", dir)
	}
	return dir
}

// TestDimsDeriveFromRealArtifact pins the derivation against the actual
// checkpoint: every value below is read from tensor shapes and config, and
// the expectations mirror the adaptive_new reference test for the same
// artifact (the port's first parity point).
func TestDimsDeriveFromRealArtifact(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": loads ~930MB weights")
	}
	model, err := Load(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	want := Dims{
		PatchLen: 32, Hidden: 1280, Layers: 20, Heads: 16, HeadDim: 80,
		Horizon: 128, Quantiles: 10, RopeTheta: 10000, RMSEps: 1e-06,
	}
	if model.Dims != want {
		t.Fatalf("derived dims = %+v, want %+v", model.Dims, want)
	}
	if len(model.Levels) != 9 {
		t.Fatalf("quantile levels = %d, want 9", len(model.Levels))
	}
	for _, name := range []string{
		"tokenizer.hidden_layer.weight",
		"stacked_xf.0.attn.qkv_proj.weight",
		"stacked_xf.19.attn.per_dim_scale.per_dim_scale",
		"output_projection_point.output_layer.weight",
	} {
		if len(model.Weights[name]) == 0 {
			t.Fatalf("tensor %q not materialized", name)
		}
	}
}
