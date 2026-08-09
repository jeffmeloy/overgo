package speechsynth

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/safetensors"
)

// TestLoadDerivesDimsAndMatchesManifest gates the loader (ladder g0): the
// artifact's tensor catalog must match the pinned manifest g0_tensors.json
// exactly (names, shapes, dtypes), and every geometric dimension must come
// out of the shapes themselves. The BF16-decoded bridge stats and BOS latent
// are gated against the g6 full-tensor dumps.
func TestLoadDerivesDimsAndMatchesManifest(t *testing.T) {
	m := loadArtifactModel(t)

	manifest := loadFixture[map[string]struct {
		DType string  `json:"dtype"`
		Shape []int64 `json:"shape"`
	}](t, "g0_tensors.json")

	source, err := safetensors.OpenSource(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if len(source.Tensors) != len(*manifest) {
		t.Fatalf("tensor count: artifact %d != manifest %d", len(source.Tensors), len(*manifest))
	}
	for name, want := range *manifest {
		tensor, ok := source.Tensors[name]
		if !ok {
			t.Fatalf("manifest tensor %q absent from artifact", name)
		}
		if tensor.DType != want.DType {
			t.Fatalf("%s dtype %s != manifest %s", name, tensor.DType, want.DType)
		}
		if len(tensor.Shape) != len(want.Shape) {
			t.Fatalf("%s shape %v != manifest %v", name, tensor.Shape, want.Shape)
		}
		for i := range want.Shape {
			if int64(tensor.Shape[i]) != want.Shape[i] {
				t.Fatalf("%s shape %v != manifest %v", name, tensor.Shape, want.Shape)
			}
		}
	}

	d := m.Dims
	t.Logf("dims: d_model=%d heads=%d head_dim=%d ff=%d layers=%d text_vocab=%d latent=%d flow_dim=%d flow_depth=%d time_freqs=%d max_period=%g",
		d.DModel, d.Heads, d.HeadDim, d.FF, d.Layers, d.TextVocab, d.LatentDim, d.FlowDim, d.FlowDepth, d.TimeFreqs, d.MaxPeriod)
	if d.DModel != d.Heads*d.HeadDim || d.LatentDim <= 0 || d.FlowDepth <= 0 {
		t.Fatalf("derived dims inconsistent: %+v", d)
	}

	g6 := loadFixture[g6Golden](t, "g6_backbone.json")
	requireWithin(t, "emb_mean (load)", m.EmbMean, g6.EmbMean.Values, tolTextEmbed)
	requireWithin(t, "emb_std (load)", m.EmbStd, g6.EmbStd.Values, tolTextEmbed)
	requireWithin(t, "bos_emb (load)", m.BosEmb, g6.BosEmb.Values, tolTextEmbed)
}

// TestLoadRefusesContradictoryConfig pins the shape-vs-config cross-check:
// a config that restates a wrong d_model must refuse, not average.
func TestLoadRefusesContradictoryConfig(t *testing.T) {
	shapes := map[string][]int{
		"flow_lm.out_norm.weight":                  {8},
		layerPrefix + "0.norm1.weight":             {8},
		layerPrefix + "0.linear1.weight":           {32, 8},
		layerPrefix + "0.self_attn.in_proj.weight": {24, 8},
		"flow_lm.conditioner.embed.weight":         {11, 8},
		"flow_lm.input_linear.weight":              {8, 4},
		"flow_lm.flow_net.input_proj.weight":       {16, 4},
		resBlockPrefix + "0.in_ln.weight":          {16},
		"flow_lm.flow_net.time_embed.0.freqs":      {4},
	}
	var config artifactConfig
	config.FlowLM.Transformer.NumHeads = 2
	config.FlowLM.Transformer.MaxPeriod = 10000
	if _, err := deriveDims(shapes, config); err != nil {
		t.Fatalf("clean derivation refused: %v", err)
	}
	config.FlowLM.Transformer.DModel = 16 // contradicts derived 8
	if _, err := deriveDims(shapes, config); err == nil {
		t.Fatal("want refusal on contradictory config d_model")
	}
}

// TestLoadArtifactPresenceContract: the model dir keeps its expected
// artifact files (config beside weights beside tokenizer).
func TestLoadArtifactPresenceContract(t *testing.T) {
	if testing.Short() {
		t.Skip("artifact stat; skipped in -short")
	}
	dir := artifactDir(t)
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s; parity NOT verified", dir)
	}
	for _, name := range []string{configFileName, "tokenizer.model"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("artifact file %s: %v", name, err)
		}
	}
}
