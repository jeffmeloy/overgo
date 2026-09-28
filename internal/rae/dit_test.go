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

// ditReference is fixtures/rae/dit.json: a seeded latent, a time and a class,
// and the velocity the reference DiT-DH predicted for them.
type ditReference struct {
	Source     string          `json:"source"`
	Commit     string          `json:"reference_commit"`
	Generator  string          `json:"generator_sha256"`
	Checkpoint string          `json:"checkpoint"`
	Seed       int             `json:"seed"`
	Time       float32         `json:"time"`
	Label      int             `json:"label"`
	Latent     referenceTensor `json:"latent"`
	Velocity   referenceTensor `json:"velocity"`
}

// TestRAEDiTVelocityMatchesReference predicts the flow velocity of the
// reference's seeded latent at its time and class with the DINOv2-B
// ImageNet-256 DiTDH-S checkpoint, and holds every value to the velocity the
// reference predicted for the same inputs.
func TestRAEDiTVelocityMatchesReference(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	fixtures := filepath.Join(root, "fixtures", "rae")
	var reference ditReference
	if err := jsonfile.DecodeStrict(filepath.Join(fixtures, "dit.json"), &reference); err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := filepath.Join(roots.ResolveModelPath("RAE-collections"), filepath.FromSlash(reference.Checkpoint))
	if _, err := os.Stat(checkpoint); err != nil {
		testskip.NotApplicable(t, "the RAE DiT checkpoint is absent: "+checkpoint)
	}
	model, err := loadDiT(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.velocity(loadReference(t, fixtures, reference.Latent), reference.Time, reference.Label)
	if err != nil {
		t.Fatal(err)
	}
	worst, scale := compareReference(t, got, loadReference(t, fixtures, reference.Velocity))
	t.Logf("max |predicted - reference| = %.3g against a largest reference value of %.3g", worst, scale)
	if worst > 1e-3*max(scale, 1) {
		t.Fatalf("velocity differs from the reference by up to %.3g", worst)
	}
}
