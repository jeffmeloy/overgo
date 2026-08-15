package oscillatorimage

import (
	"bytes"
	"image/gif"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func generatePlanar(model *Model, request Request) ([]float32, error) {
	plan, err := model.prepare(request)
	if err != nil {
		return nil, err
	}
	features, err := model.integrate(plan)
	if err != nil {
		return nil, err
	}
	image, err := model.decodePlanar(features)
	return image.Pixels, err
}

// Artifact dimensions derive from tensor extents.
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

func TestArtifactPublishesReferenceGIF(t *testing.T) {
	m := loadArtifactModel(t)
	plan, err := m.prepareVideo(VideoRequest{Class: 1, Seed: 202, Frames: 6, Scale: 8})
	if err != nil {
		t.Fatal(err)
	}
	features, err := m.integrateVideo(plan)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := m.decodeVideo(features)
	if err != nil {
		t.Fatal(err)
	}
	if encoded.MediaType != "image/gif" || encoded.Frames != 6 || encoded.Channels != 3 || encoded.Height != 64 || encoded.Width != 64 || encoded.ChangedPixels <= 0 {
		t.Fatalf("encoded video = %+v", encoded)
	}
	decoded, err := gif.DecodeAll(bytes.NewReader(encoded.Data))
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.Image) != encoded.Frames || decoded.Image[0].Bounds().Dx() != encoded.Width || decoded.Image[0].Bounds().Dy() != encoded.Height {
		t.Fatalf("decoded GIF frames=%d bounds=%v", len(decoded.Image), decoded.Image[0].Bounds())
	}
	referencePath := filepath.Join(filepath.Dir(filepath.Dir(artifactDir(t))), "docs", "image_gen_samples", "un0_video_direct_seed202_frames6.gif")
	reference, err := os.ReadFile(referencePath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: reference video %s: %v", referencePath, err)
	}
	if !bytes.Equal(encoded.Data, reference) {
		t.Fatalf("GIF differs from adaptive_new reference: bytes=%d want=%d", len(encoded.Data), len(reference))
	}
	t.Logf("real Un-0 GIF: class=1 seed=202 frames=6 size=64x64 encoded_bytes=%d changed_source_pixels=%d exact_reference=true", len(encoded.Data), encoded.ChangedPixels)
}

// Generation is deterministic and class-sensitive.
func TestArtifactGenerateDeterministicAndClassSensitive(t *testing.T) {
	m := loadArtifactModel(t)
	cfg := m.Cfg
	img, err := generatePlanar(m, Request{Class: 1, Seed: 42})
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
	again, err := generatePlanar(m, Request{Class: 1, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	other, err := generatePlanar(m, Request{Class: 2, Seed: 42})
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
	if _, err := generatePlanar(m, Request{Class: cfg.NClasses, Seed: 42}); err == nil {
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
