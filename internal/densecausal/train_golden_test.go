package densecausal

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// tinyGolden mirrors fixtures/{llama,qwen2}_train_golden.json
// (schema <arch>_train_golden/v1): a seeded tiny model, one token batch, the
// causal-LM loss, last-position logits, and every parameter gradient.
// attention_bias absent (older llama golden) reads false.
type tinyGolden struct {
	Schema string `json:"schema"`
	Config struct {
		VocabSize         int     `json:"vocab_size"`
		HiddenSize        int     `json:"hidden_size"`
		IntermediateSize  int     `json:"intermediate_size"`
		NumHiddenLayers   int     `json:"num_hidden_layers"`
		NumAttentionHeads int     `json:"num_attention_heads"`
		NumKeyValueHeads  int     `json:"num_key_value_heads"`
		HeadDim           int     `json:"head_dim"`
		RopeTheta         float64 `json:"rope_theta"`
		RMSNormEps        float64 `json:"rms_norm_eps"`
		AttentionBias     bool    `json:"attention_bias"`
	} `json:"config"`
	Tokens     []int     `json:"tokens"`
	Loss       float64   `json:"loss"`
	LastLogits []float64 `json:"last_logits"`
	Params     map[string]struct {
		Shape  []int     `json:"shape"`
		Values []float64 `json:"values"`
	} `json:"params"`
	Grads map[string][]float64 `json:"grads"`
}

func readGolden(t *testing.T, file, schema string) *tinyGolden {
	t.Helper()
	working, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(filepath.Dir(filepath.Dir(working)), "fixtures", file)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("UNAVAILABLE: %s absent; training parity NOT verified", path)
	}
	var g tinyGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Schema != schema {
		t.Fatalf("golden schema %q, want %q", g.Schema, schema)
	}
	return &g
}

func readTinyGolden(t *testing.T) *tinyGolden {
	t.Helper()
	return readGolden(t, "llama_train_golden.json", "llama_train_golden/v1")
}

func modelFromGolden(t *testing.T, g *tinyGolden) *Model {
	t.Helper()
	weights := make(map[string][]float32, len(g.Params))
	shapes := make(map[string][]int, len(g.Params))
	for name, p := range g.Params {
		values := make([]float32, len(p.Values))
		for i, v := range p.Values {
			values[i] = float32(v)
		}
		weights[name] = values
		shapes[name] = p.Shape
	}
	m, err := NewModel(weights, shapes, g.Config.NumAttentionHeads, g.Config.HeadDim, g.Config.RopeTheta, g.Config.RMSNormEps)
	if err != nil {
		t.Fatal(err)
	}
	want := Dims{
		Vocab: g.Config.VocabSize, Hidden: g.Config.HiddenSize,
		Layers: g.Config.NumHiddenLayers, Heads: g.Config.NumAttentionHeads,
		KVHeads: g.Config.NumKeyValueHeads, HeadDim: g.Config.HeadDim,
		Intermediate: g.Config.IntermediateSize,
		RopeTheta:    g.Config.RopeTheta, RMSEps: g.Config.RMSNormEps,
		AttnBias: g.Config.AttentionBias,
	}
	if m.Dims != want {
		t.Fatalf("derived dims %+v, want %+v", m.Dims, want)
	}
	return m
}

// Tolerance rationale: f32-storage host against the reference's f32 forward
// (loss compared relative), per-parameter grads absolute like the component
// goldens.
const (
	lossRelTol = 1e-6
	gradTol    = 1e-4
)

func TestTinyModelLossAndGradsMatchGolden(t *testing.T) {
	g := readTinyGolden(t)
	assertTrainGoldenParity(t, g, modelFromGolden(t, g))
}

// assertTrainGoldenParity: loss (relative), last-position logits, and every
// parameter gradient against the torch golden.
func assertTrainGoldenParity(t *testing.T, g *tinyGolden, m *Model) {
	t.Helper()
	loss, logits, grads, err := m.LossAndGrads(g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	if rel := math.Abs(loss-g.Loss) / math.Abs(g.Loss); rel > lossRelTol {
		t.Fatalf("loss %g vs golden %g (rel %g > %g)", loss, g.Loss, rel, lossRelTol)
	}
	t.Logf("loss %.9f golden %.9f", loss, g.Loss)
	last := logits[(len(g.Tokens)-1)*m.Dims.Vocab:]
	for i := range last {
		if d := math.Abs(float64(last[i]) - g.LastLogits[i]); d > gradTol {
			t.Fatalf("last_logits[%d]: |%g - %g| = %g > %g", i, last[i], g.LastLogits[i], d, gradTol)
		}
	}
	if len(grads) != len(g.Grads) {
		t.Fatalf("grad tensors %d, want %d", len(grads), len(g.Grads))
	}
	for name, want := range g.Grads {
		got, ok := grads[name]
		if !ok {
			t.Fatalf("missing grad %q", name)
		}
		if len(got) != len(want) {
			t.Fatalf("grad %q len %d, want %d", name, len(got), len(want))
		}
		maxAbs := 0.0
		for i := range got {
			if d := math.Abs(float64(got[i]) - want[i]); d > maxAbs {
				maxAbs = d
			}
		}
		if maxAbs > gradTol {
			t.Fatalf("grad %q max abs diff %g > %g", name, maxAbs, gradTol)
		}
		t.Logf("grad %-48s max abs diff %.3g", name, maxAbs)
	}
}

// TestTinyModelForwardOnlyLossAgrees ties Loss (forward-only recheck path)
// to the gradient path's loss.
func TestTinyModelForwardOnlyLossAgrees(t *testing.T) {
	g := readTinyGolden(t)
	m := modelFromGolden(t, g)
	lossForward, _, err := m.Loss(g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	lossGrad, _, _, err := m.LossAndGrads(g.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	if lossForward != lossGrad {
		t.Fatalf("forward-only loss %g != gradient-path loss %g", lossForward, lossGrad)
	}
}
