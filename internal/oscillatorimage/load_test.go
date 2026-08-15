package oscillatorimage

import (
	"bytes"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

// TestArtifactLoadDerivesDims: every geometric dim from flat tensor lengths;
// config cross-check passes on the real artifact.
func TestArtifactLoadDerivesDims(t *testing.T) {
	m := loadArtifactModel(t)
	cfg := m.Cfg
	if cfg.N != len(m.Omega) || cfg.NCond != len(m.OmegaCond) {
		t.Fatalf("frequency dims: n=%d/%d nc=%d/%d", cfg.N, len(m.Omega), cfg.NCond, len(m.OmegaCond))
	}
	if len(m.Drive) != cfg.NClasses*cfg.N*cfg.NCond {
		t.Fatalf("drive %d != %d*%d*%d", len(m.Drive), cfg.NClasses, cfg.N, cfg.NCond)
	}
	if len(m.Blocks) != len(cfg.BlockChannels) {
		t.Fatalf("blocks %d != config %d", len(m.Blocks), len(cfg.BlockChannels))
	}
	if cfg.readoutWidth() != cfg.InChannels*cfg.InH*cfg.InW {
		t.Fatalf("readout width %d != %d*%d*%d", cfg.readoutWidth(), cfg.InChannels, cfg.InH, cfg.InW)
	}
	t.Logf("namespace=%s n=%d nc=%d classes=%d in=%dx%dx%d out=%dx%dx%d steps=%d dt=%g blocks=%v",
		m.Namespace, cfg.N, cfg.NCond, cfg.NClasses, cfg.InChannels, cfg.InH, cfg.InW,
		cfg.OutChannels, cfg.OutH(), cfg.OutW(), cfg.NumSteps, cfg.Dt, cfg.BlockChannels)
}

// TestArtifactGenerateDeterministicAndClassSensitive: same seed+class is
// bit-identical; a class change must move pixels (the reference round-trip
// gates the same three facts).
func TestArtifactGenerateDeterministicAndClassSensitive(t *testing.T) {
	m := loadArtifactModel(t)
	cfg := m.Cfg
	img, err := m.Generate(1, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(img) != cfg.OutChannels*cfg.OutH()*cfg.OutW() {
		t.Fatalf("image len %d != %d*%d*%d", len(img), cfg.OutChannels, cfg.OutH(), cfg.OutW())
	}
	if cfg.TanhOut {
		for i, v := range img {
			if v < -1 || v > 1 {
				t.Fatalf("pixel %d = %g outside tanh bound", i, v)
			}
		}
	}
	again, err := m.Generate(1, 42)
	if err != nil {
		t.Fatal(err)
	}
	other, err := m.Generate(2, 42)
	if err != nil {
		t.Fatal(err)
	}
	diff := false
	for i := range img {
		if img[i] != again[i] {
			t.Fatalf("determinism diverges at %d", i)
		}
		diff = diff || other[i] != img[i]
	}
	if !diff {
		t.Fatal("class change did not change the image")
	}
	if _, err := m.Generate(cfg.NClasses, 42); err == nil {
		t.Fatal("out-of-range class accepted")
	}
}

func TestArtifactPublishesReferencePNG(t *testing.T) {
	m := loadArtifactModel(t)
	plan, err := m.prepare(Request{Class: 1, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	features, err := m.integrate(plan)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := m.decode(features)
	if err != nil {
		t.Fatal(err)
	}
	if encoded.MediaType != "image/png" || encoded.Channels != 3 || encoded.Height != m.Cfg.OutH() || encoded.Width != m.Cfg.OutW() || len(encoded.Data) == 0 {
		t.Fatalf("encoded image=%+v", encoded)
	}
	got, err := png.Decode(bytes.NewReader(encoded.Data))
	if err != nil {
		t.Fatal(err)
	}
	referencePath := filepath.Join(filepath.Dir(filepath.Dir(artifactDir(t))), "docs", "image_gen_samples", "un0_direct_seed42.png")
	referenceFile, err := os.Open(referencePath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: reference image %s: %v", referencePath, err)
	}
	want, decodeErr := png.Decode(referenceFile)
	closeErr := referenceFile.Close()
	if decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if got.Bounds() != want.Bounds() {
		t.Fatalf("PNG bounds=%v want=%v", got.Bounds(), want.Bounds())
	}
	for y := got.Bounds().Min.Y; y < got.Bounds().Max.Y; y++ {
		for x := got.Bounds().Min.X; x < got.Bounds().Max.X; x++ {
			gr, gg, gb, ga := got.At(x, y).RGBA()
			wr, wg, wb, wa := want.At(x, y).RGBA()
			if gr != wr || gg != wg || gb != wb || ga != wa {
				t.Fatalf("PNG pixel (%d,%d)=%d,%d,%d,%d want=%d,%d,%d,%d", x, y, gr, gg, gb, ga, wr, wg, wb, wa)
			}
		}
	}
	t.Logf("real Un-0 PNG: class=1 seed=42 size=%dx%d encoded_bytes=%d exact_reference_pixels=true", encoded.Width, encoded.Height, len(encoded.Data))
}
