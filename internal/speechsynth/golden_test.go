package speechsynth

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"overgo/internal/dataroot"
)

// Golden-ladder tolerances. The elementwise gates are the reference port's
// committed values (adaptive speech_flow_synthesis_test.go), each derived
// from its stage's compute depth:
//
//	1e-6  text embeddings — pure table copy of BF16-decoded weights; the
//	      reference dumps f64 of the same BF16 values, so agreement is f32
//	      rounding only.
//	1e-5  bos input row — one 32->1024 f32 linear against torch f32.
//	2e-3  transformer segments / out_norm cond / flow output — six pre-norm
//	      layers of reordered f32 GEMM accumulation.
//	1e-2  generated frame latents — closed-loop feedback re-amplifies the
//	      2e-3 open-loop divergence through input_linear each frame.
//	5e-2  EOS logits — a 1024-term dot on top of the diverged stream.
//
// Summary-trace (g3) bounds derive from the elementwise gate g of the same
// tensor: |d(rms)| <= g and |d(first_k)| <= g elementwise; |d(sum)| <= N*g.
const (
	tolTextEmbed   = 1e-6
	tolBosRow      = 1e-5
	tolTransformer = 2e-3
	tolFlowOut     = 2e-3
	tolGenLatent   = 1e-2
	tolEOSLogit    = 5e-2
)

type goldenTensor struct {
	Shape  []int     `json:"shape"`
	Values []float64 `json:"values"`
}

type goldenSummary struct {
	Shape []int     `json:"shape"`
	Numel int       `json:"numel"`
	RMS   float64   `json:"rms"`
	Sum   float64   `json:"sum"`
	First []float64 `json:"first"`
}

type g6Golden struct {
	Text              string       `json:"text"`
	TextEmbeddings    goldenTensor `json:"text_embeddings"`
	VoiceConditioning goldenTensor `json:"voice_conditioning"`
	TransformerCalls  []struct {
		In  goldenTensor `json:"in"`
		Out goldenTensor `json:"out"`
	} `json:"transformer_calls"`
	FlowStep struct {
		Cond    goldenTensor `json:"cond"`
		S, T    float64
		NoiseIn goldenTensor `json:"noise_in"`
		FlowOut goldenTensor `json:"flow_out"`
	} `json:"flow_step"`
	EmbMean goldenTensor `json:"emb_mean"`
	EmbStd  goldenTensor `json:"emb_std"`
	BosEmb  goldenTensor `json:"bos_emb"`
}

type g7Golden struct {
	FlowCalls []struct {
		Cond  goldenTensor `json:"cond"`
		S, T  float64
		Noise goldenTensor `json:"noise"`
		Out   goldenTensor `json:"out"`
	} `json:"flow_calls"`
	NQuantizerCalls int          `json:"n_quantizer_calls"`
	EOSLogits       []float64    `json:"eos_logits"`
	PCM             goldenTensor `json:"pcm"`
}

type g3Golden struct {
	Traces map[string]goldenSummary `json:"traces"`
}

func repoRoot(t *testing.T) string {
	t.Helper()
	working, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Dir(filepath.Dir(working)) // internal/speechsynth -> repo root
}

func fixturePath(t *testing.T, name string) string {
	t.Helper()
	return filepath.Join(repoRoot(t), "fixtures", "pockettts", name)
}

// loadFixture decodes a committed golden; absence skips LOUDLY.
func loadFixture[T any](t *testing.T, name string) *T {
	t.Helper()
	raw, err := os.ReadFile(fixturePath(t, name))
	if err != nil {
		t.Skipf("UNAVAILABLE: golden %s absent; parity NOT verified", name)
	}
	out := new(T)
	if err := json.Unmarshal(raw, out); err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return out
}

func f32of(values []float64) []float32 {
	out := make([]float32, len(values))
	for i, v := range values {
		out[i] = float32(v)
	}
	return out
}

func maxAbsDiff(got []float32, want []float64) float64 {
	worst := 0.0
	for i := range got {
		if d := math.Abs(float64(got[i]) - want[i]); d > worst {
			worst = d
		}
	}
	return worst
}

// requireWithin reports the measured max abs diff for the parity ledger and
// fails past the gate.
func requireWithin(t *testing.T, name string, got []float32, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != golden %d", name, len(got), len(want))
	}
	diff := maxAbsDiff(got, want)
	t.Logf("%s: max abs diff %.6e (gate %.0e)", name, diff, tol)
	if diff > tol {
		t.Fatalf("%s diverges: %g > %g", name, diff, tol)
	}
}

// Shared once-loaded artifact for the real-model gates.
var (
	modelOnce sync.Once
	modelInst *Model
	modelErr  error
)

func loadArtifactModel(t *testing.T) *Model {
	t.Helper()
	if testing.Short() {
		t.Skip("loads the full artifact; skipped in -short")
	}
	roots, err := dataroot.Resolve(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(roots.Models, "pocket-tts")
	if _, err := os.Stat(filepath.Join(dir, configFileName)); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s; parity NOT verified", dir)
	}
	modelOnce.Do(func() { modelInst, modelErr = Load(dir) })
	if modelErr != nil {
		t.Fatal(modelErr)
	}
	return modelInst
}

func artifactDir(t *testing.T) string {
	t.Helper()
	roots, err := dataroot.Resolve(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(roots.Models, "pocket-tts")
}
