package latentvideo

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testutil"
)

type g1Tensor struct {
	File     string  `json:"file"`
	Shape    []int   `json:"shape"`
	Elements int     `json:"elements"`
	SHA256   string  `json:"sha256"`
	Min      float64 `json:"min"`
	Max      float64 `json:"max"`
}

type g1Branch struct {
	Stats struct {
		TokenCount    int    `json:"token_count"`
		EncoderLayers int    `json:"encoder_layers"`
		Engine        string `json:"encoder_engine"`
	} `json:"stats"`
	Tensor g1Tensor `json:"tensor"`
}

type g1Golden struct {
	Conditional    g1Branch `json:"conditional"`
	Unconditional  g1Branch `json:"unconditional"`
	Dim            int      `json:"dim"`
	TextLen        int      `json:"text_len"`
	Prompt         string   `json:"prompt"`
	NegativePrompt string   `json:"negative_prompt"`
}

func loadG1(t *testing.T) (g1Golden, string) {
	t.Helper()
	dir := testutil.FixturePath(t, "wan")
	raw, err := os.ReadFile(filepath.Join(dir, "g1_text_conditioning.json"))
	if err != nil {
		t.Fatalf("missing required golden manifest: %v", err)
	}
	var g g1Golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g, dir
}

func loadG1Tensor(t *testing.T, dir string, spec g1Tensor) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(spec.File)))
	if err != nil {
		t.Fatalf("missing required golden tensor: %v", err)
	}
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != spec.SHA256 {
		t.Fatalf("golden tensor %s sha256=%s want %s", spec.File, got, spec.SHA256)
	}
	if len(raw) != spec.Elements*4 {
		t.Fatalf("golden tensor %s bytes=%d want %d", spec.File, len(raw), spec.Elements*4)
	}
	out := make([]float32, spec.Elements)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func wanModelDir(t testing.TB) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("UNAVAILABLE: model dir absent at %s: %v", dir, err)
	}
	return dir
}

// Cross-engine gates: the golden context was captured on the CUDA BF16
// engine; this port is the host BF16 path. Host fidelity is proven
// separately at tolerance 0 (encoder_real_test.go bit-parity vs adaptive's
// current host path), so residual disagreement here is purely accumulation
// order between engines whose op outputs both round to BF16, compounding
// over 24 layers. Measured 2026-08-09: pad rows (projection of a zero
// encoder row — no encoder in the path) agree at 7.6e-6 on both branches;
// token rows carry the cross-engine noise, worst-element 2.2465e-2 /
// mean 8.43e-5 (cond, 22 tokens) and 4.1992e-2 / 4.65e-4 (uncond, 126
// tokens). The reference repo's own host-vs-CUDA gates on this artifact are
// 6e-3 sampled elements / 1e-2 max-abs statistic for a 3-token prompt.
// Worst-element gate: measured worst + ~40% margin; mean gate: measured
// worst x2 — the mean is the strong signal (a real port defect moves it by
// orders, per the stale-constant diagnosis runs).
const (
	goldenCrossEngineMaxTolerance  = 6e-2
	goldenCrossEngineMeanTolerance = 1e-3
)

// TestTextConditioningGoldenParity: full pipeline vs the committed g1 cond
// and uncond contexts (fox prompt / negative prompt).
func TestTextConditioningGoldenParity(t *testing.T) {
	if testing.Short() {
		t.Skip("streams the 11.4GB encoder checkpoint twice; skipped in -short")
	}
	g, fixtureDir := loadG1(t)
	modelDir := wanModelDir(t)
	spec := TextConditioningSpec{
		TokenizerDir:      filepath.Join(modelDir, "google", "umt5-xxl"),
		EncoderCheckpoint: filepath.Join(modelDir, "models_t5_umt5-xxl-enc-bf16.pth"),
		ProjectionDir:     modelDir,
		SequenceLength:    g.TextLen,
		// Published encoder-config facts (relative_attention_max_distance,
		// layer_norm_epsilon) — the checkpoint carries no config JSON.
		RelativeMaxDistance: referenceEncoderPolicy.RelativeMaxDistance,
		NormEps:             referenceEncoderPolicy.NormEps,
	}
	if _, err := os.Stat(spec.EncoderCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: encoder checkpoint absent: %v", err)
	}
	for _, branch := range []struct {
		name   string
		prompt string
		golden g1Branch
	}{
		{"conditional", g.Prompt, g.Conditional},
		{"unconditional", g.NegativePrompt, g.Unconditional},
	} {
		want := loadG1Tensor(t, fixtureDir, branch.golden.Tensor)
		got, err := TextConditioning(spec, branch.prompt)
		if err != nil {
			t.Fatalf("%s: %v", branch.name, err)
		}
		if got.Rows != g.TextLen || got.Dim != g.Dim || len(got.Context) != len(want) {
			t.Fatalf("%s shape rows=%d dim=%d len=%d want %dx%d", branch.name, got.Rows, got.Dim, len(got.Context), g.TextLen, g.Dim)
		}
		if got.TokenCount != branch.golden.Stats.TokenCount {
			t.Fatalf("%s token count=%d want %d", branch.name, got.TokenCount, branch.golden.Stats.TokenCount)
		}
		if got.Encoder.Layers != branch.golden.Stats.EncoderLayers {
			t.Fatalf("%s encoder layers=%d want %d", branch.name, got.Encoder.Layers, branch.golden.Stats.EncoderLayers)
		}
		// Streaming evidence: whole-checkpoint residency would show as peak
		// heap at checkpoint scale (11.4GB); the bound is one F32-decoded
		// block plus activations.
		if got.Encoder.MaxBlockWeightBytes <= 0 || got.Encoder.BlockWeightBytes <= got.Encoder.MaxBlockWeightBytes {
			t.Fatalf("%s streaming stats degenerate: %+v", branch.name, got.Encoder)
		}
		if got.Encoder.PeakHeapAllocBytes > 4<<30 {
			t.Fatalf("%s peak heap %d exceeds streamed bound", branch.name, got.Encoder.PeakHeapAllocBytes)
		}
		var maxDiff, sumDiff, maxTokenDiff, maxPadDiff float64
		for i := range want {
			d := math.Abs(float64(got.Context[i]) - float64(want[i]))
			sumDiff += d
			if d > maxDiff {
				maxDiff = d
			}
			if i < got.TokenCount*g.Dim {
				if d > maxTokenDiff {
					maxTokenDiff = d
				}
			} else if d > maxPadDiff {
				maxPadDiff = d
			}
		}
		meanDiff := sumDiff / float64(len(want))
		t.Logf("%s tokens=%d engine=%s wall=%.1fs max_abs_diff=%.6g (token_rows=%.6g pad_rows=%.6g) mean_abs_diff=%.6g peak_heap_alloc_bytes=%d max_block_weight_bytes=%d total_block_weight_bytes=%d",
			branch.name, got.TokenCount, got.Encoder.Engine, got.Encoder.EncoderWallSec, maxDiff, maxTokenDiff, maxPadDiff, meanDiff, got.Encoder.PeakHeapAllocBytes, got.Encoder.MaxBlockWeightBytes, got.Encoder.BlockWeightBytes)
		if maxDiff > goldenCrossEngineMaxTolerance {
			t.Fatalf("%s max abs diff=%g > %g", branch.name, maxDiff, goldenCrossEngineMaxTolerance)
		}
		if meanDiff > goldenCrossEngineMeanTolerance {
			t.Fatalf("%s mean abs diff=%g > %g", branch.name, meanDiff, goldenCrossEngineMeanTolerance)
		}
	}
}
