package densecausaltrain

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/densecausal"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// tinyMuonModel is the seeded two-layer model densecausal's host trainer
// is held to.
func tinyMuonModel(t *testing.T) *densecausal.Model {
	t.Helper()
	spec := testutil.DenseCausalSpec{
		Vocab: 32, Hidden: 16, Heads: 2, HeadDim: 8,
		KVHeads: 2, Intermediate: 32, Layers: 2, Seed: 1,
	}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	m, err := densecausal.NewModel(weights, shapes, spec.Heads, spec.HeadDim, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// derivedTestLearningRate is the builtin policy's rate for the model's
// parameter count.
func derivedTestLearningRate(t *testing.T, model *densecausal.Model) float64 {
	t.Helper()
	plan, err := model.TrainingPlan()
	if err != nil {
		t.Fatal(err)
	}
	return trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(plan.ParameterCount())
}

// qwen2Golden is the part of fixtures/qwen2_train_golden.json the resident
// trainer is compared on: the seeded tiny model with attention biases and
// its token batch.
type qwen2Golden struct {
	Schema string `json:"schema"`
	Config struct {
		NumAttentionHeads int     `json:"num_attention_heads"`
		HeadDim           int     `json:"head_dim"`
		RopeTheta         float64 `json:"rope_theta"`
		RMSNormEps        float64 `json:"rms_norm_eps"`
	} `json:"config"`
	Tokens []int `json:"tokens"`
	Params map[string]struct {
		Shape  []int     `json:"shape"`
		Values []float64 `json:"values"`
	} `json:"params"`
}

func readQwen2Golden(t *testing.T) *qwen2Golden {
	t.Helper()
	path := testutil.FixturePath(t, "qwen2_train_golden.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("UNAVAILABLE: %s absent; training parity NOT verified", path)
	}
	var g qwen2Golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Schema != "qwen2_train_golden/v1" {
		t.Fatalf("golden schema %q, want qwen2_train_golden/v1", g.Schema)
	}
	return &g
}

func modelFromGolden(t *testing.T, g *qwen2Golden) *densecausal.Model {
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
	m, err := densecausal.NewModel(weights, shapes, g.Config.NumAttentionHeads, g.Config.HeadDim, g.Config.RopeTheta, g.Config.RMSNormEps)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Dims.AttnBias {
		t.Fatal("qwen2 golden must derive AttnBias=true")
	}
	return m
}

// artifactDir resolves a real artifact through the data-root contract;
// absent artifact skips LOUDLY.
func artifactDir(t *testing.T, name string) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(roots.Models, name)
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		if _, err := os.Stat(filepath.Join(dir, "model.safetensors.index.json")); err != nil {
			t.Skipf("UNAVAILABLE: %s artifact absent at %s; real-artifact training NOT verified", name, dir)
		}
	}
	return dir
}
