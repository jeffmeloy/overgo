package diffusionimage

import (
	"math"
	"testing"
)

// TestRealCheckpointDerivedConfig: the derivation must reconstruct the
// vendor architecture (train.py AsymmetricResidualUDiT arguments) from the
// checkpoint alone.
func TestRealCheckpointDerivedConfig(t *testing.T) {
	m := loadArtifactModel(t)
	want := Config{
		InChannels: 3, BaseChannels: 128, PatchSize: 4, NumLevels: 3,
		MidBlocks: 16, Groups: 32, Heads: 8, QRank: 6, KVRank: 2,
		NormEps: 1e-5, RopeTheta: 10000,
	}
	if m.Cfg != want {
		t.Fatalf("derived config %+v, want %+v", m.Cfg, want)
	}
	// Block census: this checkpoint carries ResBlocks at every level
	// (3 encoder / 7 decoder per level) and 16 middle transformers.
	for _, family := range []struct {
		name   string
		levels []level
		blocks int
	}{{"encoders", m.encoders, 3}, {"decoders", m.decoders, 7}} {
		if len(family.levels) != 3 {
			t.Fatalf("%s levels %d, want 3", family.name, len(family.levels))
		}
		for i := range family.levels {
			if got := len(family.levels[i].blocks); got != family.blocks {
				t.Fatalf("%s level %d blocks %d, want %d", family.name, i, got, family.blocks)
			}
			for _, blk := range family.levels[i].blocks {
				if _, ok := blk.(*resBlock); !ok {
					t.Fatalf("%s level %d holds a non-ResBlock", family.name, i)
				}
			}
		}
	}
}

// TestRealCheckpointVendorParity: full forward on the real artifact vs the
// vendor-script golden (torch strict state-dict load, 1x3x32x32). Loads
// FRESH per invocation — the matched performance protocol times this test
// -count=5 against the adaptive twin, whose helper also loads per test.
func TestRealCheckpointVendorParity(t *testing.T) {
	requireLongTest(t)
	g := loadGolden[struct {
		B, C, H, W int
		X, Out     []float32
	}](t, "real_forward")
	m, err := Load(artifactDir(t))
	if err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", artifactDir(t), err)
	}
	got, err := m.Forward(g.X, g.B, g.H, g.W)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "real forward", got, g.Out, tolReal)
}

// TestRealCheckpointSample: the generate profile's execution (Euler flow
// from seeded noise) stays finite and non-degenerate on the real weights.
func TestRealCheckpointSample(t *testing.T) {
	m := loadArtifactModel(t)
	const steps = 4
	img, err := m.Sample(t.Context(), 1, 32, 32, steps, 42)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, v := range img {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("nonfinite sample output")
		}
		sum += math.Abs(float64(v))
	}
	if sum == 0 {
		t.Fatal("all-zero sample output")
	}
	t.Logf("sample: %d px, mean |x| %.4f after %d Euler steps", len(img), sum/float64(len(img)), steps)
}
