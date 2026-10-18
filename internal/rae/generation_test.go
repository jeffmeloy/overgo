package rae

import (
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testutil"
	"overgo/internal/torchrng"
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

// TestSeededNoiseMatchesReference draws the generation golden's seed as the
// reference's CPU torch generator does and holds every value to the noise
// the reference sampled from.
func TestSeededNoiseMatchesReference(t *testing.T) {
	t.Parallel()
	fixtures := filepath.Join(testutil.RepoRoot(t), "fixtures", "rae")
	var reference generationReference
	if err := jsonfile.DecodeStrict(filepath.Join(fixtures, "generation.json"), &reference); err != nil {
		t.Fatal(err)
	}
	want := loadReference(t, fixtures, reference.Noise)
	got, err := torchrng.CPUNormal(int64(reference.Seed), len(want))
	if err != nil {
		t.Fatal(err)
	}
	worst, scale := compareReference(t, got, want)
	t.Logf("max |drawn - reference| = %.3g against a largest reference value of %.3g", worst, scale)
	if worst > 1e-6*scale {
		t.Fatalf("seeded noise differs from the reference by up to %.3g", worst)
	}
	if _, err := torchrng.CPUNormal(int64(reference.Seed), 8); err == nil {
		t.Fatal("a draw shorter than torch's fill was accepted")
	}
}
