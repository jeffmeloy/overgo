package speechsynth

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/media"
)

// Codec-ladder tolerances: the reference port's committed gates
// (adaptive speech_flow_synthesis_test.go / research_test.go), each derived
// from its stage's compute depth:
//
//	1e-4  upsample — one depthwise k=32 transpose conv of BF16 weights.
//	5e-4  decoder transformer — two pre-norm layers of reordered f32 GEMMs.
//	1e-3  SEANet decode / decode-from-latent PCM — the deep conv stack.
//	2e-2  generated e2e PCM — closed-loop latent divergence (1e-2 per frame
//	      gate) pushed through the quantizer projection and codec.
//	1e-3  PCM rms — energy statistic of the same tensors.
//
// The reference-wav gate is a triangle bound: this port and the adaptive
// port each gate 2e-2 elementwise against the SAME torch reference PCM, and
// the committed wav quantizes adaptive's samples to PCM16 (+2^-15).
const (
	tolUpsample    = 1e-4
	tolCodecTr     = 5e-4
	tolSeanetPCM   = 1e-3
	tolE2EPCM      = 2e-2
	tolPCMRMS      = 1e-3
	tolRefWavPCM   = 2*tolE2EPCM + 1.0/(1<<15)
	e2eTempWavName = "overgo_pockettts_e2e.wav"
)

type g5Golden struct {
	LatentIn goldenTensor `json:"latent_in"`
	Stages   map[string]struct {
		In  *goldenTensor `json:"in"`
		Out goldenTensor  `json:"out"`
	} `json:"stages"`
	PCM goldenSummary `json:"pcm"`
}

func pcmRMS(pcm []float32) float64 {
	var rms float64
	for _, v := range pcm {
		rms += float64(v) * float64(v)
	}
	return math.Sqrt(rms / float64(len(pcm)))
}

// TestMimiDecodeStagesGolden gates each codec decode stage against the full
// reference boundary tensors (ladder g5), then latent -> PCM end-to-end.
func TestMimiDecodeStagesGolden(t *testing.T) {
	m := loadArtifactModel(t)
	c := m.Codec
	g := loadFixture[g5Golden](t, "g5_mimi_decode_stages.json")

	if len(g.LatentIn.Shape) != 3 || g.LatentIn.Shape[1] != c.outerDim {
		t.Fatalf("unexpected latent shape %v", g.LatentIn.Shape)
	}
	T := g.LatentIn.Shape[2]
	latent := f32of(g.LatentIn.Values)

	// Stage 1: depthwise transpose-conv upsample [outer,T] -> [outer,T*16].
	up := c.Upsample(latent, T)
	upStage := g.Stages["upsample"]
	upT := upStage.Out.Shape[2]
	if len(up) != c.outerDim*upT {
		t.Fatalf("upsample length %d want %d", len(up), c.outerDim*upT)
	}
	requireWithin(t, "upsample", up, upStage.Out.Values, tolUpsample)

	// Stage 2: windowed decoder transformer; the reference records its
	// output as the SEANet decoder stage's input tensor.
	decStage := g.Stages["decoder"]
	if decStage.In == nil {
		t.Fatal("g5 decoder stage missing input tensor")
	}
	c.TransformInPlace(up, upT)
	requireWithin(t, "decoder transformer", up, decStage.In.Values, tolCodecTr)

	// Stage 3: SEANet conv stack -> PCM, from the golden stage input.
	pcm := c.SeanetDecode(f32of(decStage.In.Values), upT)
	if len(pcm) != decStage.Out.Shape[2] {
		t.Fatalf("pcm length %d want %d", len(pcm), decStage.Out.Shape[2])
	}
	requireWithin(t, "seanet decode", pcm, decStage.Out.Values, tolSeanetPCM)

	// End-to-end: latent -> pcm, gated on the pcm head slice + rms.
	full := c.DecodeFromLatent(latent, T)
	if len(full) != g.PCM.Numel {
		t.Fatalf("e2e length %d want %d", len(full), g.PCM.Numel)
	}
	requireWithin(t, "decode-from-latent pcm slice", full[:len(g.PCM.First)], g.PCM.First, tolSeanetPCM)
	if rms := pcmRMS(full); math.Abs(rms-g.PCM.RMS) > tolPCMRMS {
		t.Fatalf("e2e rms %g want %g", rms, g.PCM.RMS)
	} else {
		t.Logf("decode-from-latent rms %.8f (golden %.8f, gate %.0e)", rms, g.PCM.RMS, tolPCMRMS)
	}
}

// TestSpeakE2EProducesReferenceAudio is the codec-boundary golden: the full
// pipeline text tokens + voice conditioning -> latents -> REAL PCM audio
// gated against the reference generation (g7 torch PCM), the e2e summary
// (g4), and the committed reference WAV — the rung-1 pattern where the
// boundary refusal became the golden output test. The synthesized audio is
// written as a real WAV to the OS temp dir.
func TestSpeakE2EProducesReferenceAudio(t *testing.T) {
	m := loadArtifactModel(t)
	g6 := loadFixture[g6Golden](t, "g6_backbone.json")
	g7 := loadFixture[g7Golden](t, "g7_generation.json")
	g1 := loadFixture[struct {
		IDs []int `json:"ids"`
	}](t, "g1_tokenizer.json")
	g4 := loadFixture[struct {
		SampleRate int           `json:"sample_rate"`
		PCM        goldenSummary `json:"pcm"`
	}](t, "g4_e2e.json")

	voiceCond := f32of(g6.TransformerCalls[0].In.Values)
	tv := g6.TransformerCalls[0].In.Shape[1]
	nFrames := g7.NQuantizerCalls
	noises := make([][]float32, nFrames)
	for i := range noises {
		noises[i] = f32of(g7.FlowCalls[2+i].Noise.Values)
	}

	wallStart := time.Now()
	latents, _, err := m.GenerateLatents(voiceCond, tv, g1.IDs, GenerateParams{
		MaxFrames:    nFrames,
		EOSThreshold: math.Inf(1),
		NoiseAt:      func(step int, dst []float32) { copy(dst, noises[step]) },
	})
	if err != nil {
		t.Fatal(err)
	}
	backboneWall := time.Since(wallStart)
	codecStart := time.Now()
	pcm, err := m.LatentsToPCM(latents)
	if err != nil {
		t.Fatal(err)
	}
	codecWall := time.Since(codecStart)

	if len(pcm) != len(g7.PCM.Values) {
		t.Fatalf("pcm length %d want %d", len(pcm), len(g7.PCM.Values))
	}
	requireWithin(t, "e2e pcm vs g7", pcm, g7.PCM.Values, tolE2EPCM)
	rms := pcmRMS(pcm)
	wantRMS := pcmRMS(f32of(g7.PCM.Values))
	if math.Abs(rms-wantRMS) > tolPCMRMS {
		t.Fatalf("pcm rms %g want %g", rms, wantRMS)
	}

	// g4 e2e summary: same reference run's committed numel/rms/head slice.
	if g4.SampleRate != m.Codec.SampleRate {
		t.Fatalf("g4 sample rate %d != codec %d", g4.SampleRate, m.Codec.SampleRate)
	}
	if len(pcm) != g4.PCM.Numel {
		t.Fatalf("pcm length %d != g4 numel %d", len(pcm), g4.PCM.Numel)
	}
	requireWithin(t, "e2e pcm head vs g4", pcm[:len(g4.PCM.First)], g4.PCM.First, tolE2EPCM)
	if math.Abs(rms-g4.PCM.RMS) > tolPCMRMS {
		t.Fatalf("pcm rms %g vs g4 %g", rms, g4.PCM.RMS)
	}

	// Committed reference WAV: adaptive's generated audio, PCM16-quantized.
	raw, err := os.ReadFile(fixturePath(t, "_go_generated.wav"))
	if err != nil {
		t.Skipf("UNAVAILABLE: reference wav absent; e2e audio parity NOT verified: %v", err)
	}
	refPCM, refRate, err := media.DecodeWAV(raw)
	if err != nil {
		t.Fatal(err)
	}
	if refRate != m.Codec.SampleRate {
		t.Fatalf("reference wav rate %d != codec %d", refRate, m.Codec.SampleRate)
	}
	if len(refPCM) != len(pcm) {
		t.Fatalf("reference wav %d samples != generated %d", len(refPCM), len(pcm))
	}
	refF64 := make([]float64, len(refPCM))
	for i, v := range refPCM {
		refF64[i] = float64(v)
	}
	requireWithin(t, "e2e pcm vs reference wav", pcm, refF64, tolRefWavPCM)

	// Real audio artifact: encode + write the synthesized WAV.
	encoded, err := media.EncodeWAVPCM16(pcm, m.Codec.SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	outPath := filepath.Join(os.TempDir(), e2eTempWavName)
	if err := os.WriteFile(outPath, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	seconds := float64(len(pcm)) / float64(m.Codec.SampleRate)
	t.Logf("e2e audio: %s (%d bytes), %.3fs @ %d Hz, rms %.6f; wall backbone %.2fs codec %.2fs",
		outPath, len(encoded), seconds, m.Codec.SampleRate, rms,
		backboneWall.Seconds(), codecWall.Seconds())
}

// TestLatentsToPCMContracts pins the named refusals that remain at the
// codec boundary now that the decoder is real: no loaded codec, and
// incompatible latent geometry.
func TestLatentsToPCMContracts(t *testing.T) {
	var unloaded Model
	if pcm, err := unloaded.LatentsToPCM(LatentBatch{Values: make([]float32, 32), Frames: 1, Width: 32}); err == nil || pcm != nil {
		t.Fatal("want refusal without a loaded codec")
	}
	m := loadArtifactModel(t)
	if _, err := m.LatentsToPCM(LatentBatch{Values: make([]float32, 8), Frames: 1, Width: 8}); err == nil {
		t.Fatal("want refusal on latent width mismatch")
	}
	if _, err := m.LatentsToPCM(LatentBatch{Values: nil, Frames: 0, Width: m.Dims.LatentDim}); err == nil {
		t.Fatal("want refusal on empty batch")
	}
}

// TestResamplePolyMatchesSciPy gates the input-side polyphase resampler
// against the scipy golden (ladder g9). Bound: each output is one FIR dot
// product over the fixture's taps in f32 against the f64 golden — four
// sigma of ulp32*sqrt(taps) accumulation rounding on a unit-scale signal
// (the reference port's accumParityF32Tol derivation).
func TestResamplePolyMatchesSciPy(t *testing.T) {
	g := loadFixture[struct {
		Up   int       `json:"up"`
		Down int       `json:"down"`
		Taps []float64 `json:"taps"`
		X    []float64 `json:"x"`
		Y    []float64 `json:"y"`
	}](t, "g9_resample.json")
	got, err := media.ResamplePoly(t.Context(), nil, f32of(g.X), g.Up, g.Down, g.Taps)
	if err != nil {
		t.Fatal(err)
	}
	const ulp32 = 1.0 / (1 << 23)
	tol := 4 * ulp32 * math.Sqrt(float64(len(g.Taps)))
	requireWithin(t, "resample_poly", got, g.Y, tol)
}
