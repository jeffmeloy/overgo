package oscillatorimage

import "testing"

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
