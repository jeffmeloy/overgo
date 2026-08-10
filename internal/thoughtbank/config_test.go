package thoughtbank

import (
	"os"
	"testing"

	"overgo/internal/pytorchzip"
)

// defaultModelPath is where the shipped Fractale-350M-base checkpoint lives;
// override with THOUGHTBANK_MODEL_PT. The test skips when the file is absent, so
// it is model-gated rather than a hard dependency.
const defaultModelPath = `C:\Users\jeffm\adaptive_new\models\Fractale-350M-base\model.pt`

func modelPath() string {
	if p := os.Getenv("THOUGHTBANK_MODEL_PT"); p != "" {
		return p
	}
	return defaultModelPath
}

func cfgInt(t *testing.T, cfg map[string]any, key string) int {
	t.Helper()
	v, ok := cfg[key]
	if !ok {
		t.Fatalf("ck[cfg] missing %q", key)
	}
	switch n := v.(type) {
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		t.Fatalf("ck[cfg][%q] is %T, not numeric", key, v)
		return 0
	}
}

// TestConfigParseMatchesShippedCheckpoint reads the real 386M-parameter
// Fractale-350M-base model.pt through pytorchzip, derives the architecture from
// the tensor inventory, and asserts (a) the derived shape-dims equal the shipped
// 12L/768 configuration, (b) they agree with the pickled ck["cfg"], and (c) the
// tensor inventory is complete: every tensor the forward binds is present and no
// tensor is unaccounted. No torch and no tokenizer download are involved.
func TestConfigParseMatchesShippedCheckpoint(t *testing.T) {
	path := modelPath()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("checkpoint unavailable at %s: %v", path, err)
	}

	metas, err := pytorchzip.ReadTensorMetadata(path)
	if err != nil {
		t.Fatalf("ReadTensorMetadata: %v", err)
	}
	shapes := make(map[string][]int64, len(metas))
	for _, m := range metas {
		shapes[m.Name] = m.Shape
	}

	cfg, err := pytorchzip.ReadScalarConfig(path)
	if err != nil {
		t.Fatalf("ReadScalarConfig: %v", err)
	}
	swiGLU, _ := cfg["mem_read_swiglu"].(bool)
	if !swiGLU {
		t.Fatalf("ck[cfg].mem_read_swiglu = %v, want true for the shipped checkpoint", cfg["mem_read_swiglu"])
	}

	c, err := DeriveArchFromShapes(shapes, swiGLU)
	if err != nil {
		t.Fatalf("DeriveArchFromShapes: %v", err)
	}

	// (a) Derived shape-dims equal the shipped 386M configuration.
	for _, w := range []struct {
		name     string
		got, exp int
	}{
		{"vocab", c.Vocab, 49154}, {"d_model", c.DModel, 768},
		{"n_layers", c.NLayers, 12}, {"n_heads", c.NHeads, 12},
		{"d_head", c.DHead, 64}, {"n_hc", c.NHC, 2},
		{"n_groups", c.NGroups, 2}, {"n_idx_heads", c.NIdxHeads, 3},
		{"d_latent_q", c.DLatentQ, 64}, {"csa_m", c.CSAm, 4}, {"hca_m", c.HCAm, 16},
		{"n_experts", c.NExperts, 4}, {"n_shared", c.NShared, 1},
		{"d_ff", c.DFF, 1536}, {"mem_dim", c.MemDim, 512}, {"mem_read_rank", c.MemReadRank, 8},
	} {
		if w.got != w.exp {
			t.Errorf("derived %s = %d, shipped want %d", w.name, w.got, w.exp)
		}
	}
	if !c.Untied {
		t.Errorf("expected an untied lm_head.weight in the shipped checkpoint")
	}

	// (b) Shape-derived dims agree with the pickled ck["cfg"] (independent source).
	for _, w := range []struct {
		key string
		got int
	}{
		{"vocab_size", c.Vocab}, {"d_model", c.DModel}, {"n_layers", c.NLayers},
		{"n_heads", c.NHeads}, {"d_head", c.DHead}, {"n_hc", c.NHC},
		{"n_groups", c.NGroups}, {"d_latent_q", c.DLatentQ},
		{"csa_m", c.CSAm}, {"hca_m", c.HCAm},
		{"n_experts", c.NExperts}, {"n_shared", c.NShared},
		{"d_ff", c.DFF}, {"mem_dim", c.MemDim}, {"mem_read_rank", c.MemReadRank},
	} {
		if declared := cfgInt(t, cfg, w.key); declared != w.got {
			t.Errorf("ck[cfg].%s = %d but shape-derived = %d", w.key, declared, w.got)
		}
	}

	// Behavioural fields come only from ck[cfg]; record and sanity-check them.
	c.MaxMem = cfgInt(t, cfg, "max_mem")
	c.TopKCSA = cfgInt(t, cfg, "top_k_csa")
	c.TopKExperts = cfgInt(t, cfg, "top_k_experts")
	c.NWin = cfgInt(t, cfg, "n_win")
	c.SinkhornIters = cfgInt(t, cfg, "sinkhorn_iters")
	c.MemSeedSlots = cfgInt(t, cfg, "mem_seed_slots")
	for _, w := range []struct {
		name     string
		got, exp int
	}{
		{"max_mem", c.MaxMem, 8}, {"top_k_csa", c.TopKCSA, 8},
		{"top_k_experts", c.TopKExperts, 2}, {"n_win", c.NWin, 16},
		{"sinkhorn_iters", c.SinkhornIters, 20}, {"mem_seed_slots", c.MemSeedSlots, 4},
	} {
		if w.got != w.exp {
			t.Errorf("ck[cfg].%s = %d, want %d", w.name, w.got, w.exp)
		}
	}

	// (c) Inventory completeness: expected == present, both directions.
	expected := ExpectedTensorNames(c)
	expSet := make(map[string]bool, len(expected))
	for _, n := range expected {
		expSet[n] = true
		if _, ok := shapes[n]; !ok {
			t.Errorf("expected tensor %q missing from checkpoint", n)
		}
	}
	for name := range shapes {
		if !expSet[name] {
			t.Errorf("checkpoint tensor %q is unaccounted (no binding site)", name)
		}
	}
	if len(expected) != len(shapes) {
		t.Errorf("tensor count mismatch: expected %d, checkpoint has %d", len(expected), len(shapes))
	}
	t.Logf("config-parse OK: %d tensors, %d layers, vocab=%d d_model=%d, all shape-dims agree with ck[cfg]",
		len(shapes), c.NLayers, c.Vocab, c.DModel)
}
