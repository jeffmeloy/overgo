package composition

import (
	"testing"

	"overgo/internal/organ"
	"overgo/internal/testutil"
)

// TestViabilitySelectsOrganClassifiedComponent pins the generalized donor
// selection: components come from organ classification over the weight
// inventory, never hardcoded family names. HF and GGUF conventions both
// classify; ordering is deterministic; incomplete triples and out-of-range
// indices are refused by name.
func TestViabilitySelectsOrganClassifiedComponent(t *testing.T) {
	weights, _ := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 16, Hidden: 8, Heads: 2, HeadDim: 4,
		KVHeads: 2, Intermediate: 16, Layers: 3, Seed: 5,
	})
	gate, up, down, contract, err := SelectDonorMLP(weights, 1)
	if err != nil {
		t.Fatal(err)
	}
	if gate != "model.layers.1.mlp.gate_proj.weight" || up != "model.layers.1.mlp.up_proj.weight" ||
		down != "model.layers.1.mlp.down_proj.weight" || contract.Role != organ.RoleMLPGate {
		t.Fatalf("HF selection = (%s, %s, %s, %s)", gate, up, down, contract.Role)
	}

	gguf := map[string][]float32{
		"blk.0.ffn_gate.weight": nil, "blk.0.ffn_up.weight": nil, "blk.0.ffn_down.weight": nil,
		"blk.0.attn_q.weight": nil, "token_embd.weight": nil,
	}
	gate, up, down, _, err = SelectDonorMLP(gguf, 0)
	if err != nil || gate != "blk.0.ffn_gate.weight" || up != "blk.0.ffn_up.weight" || down != "blk.0.ffn_down.weight" {
		t.Fatalf("GGUF selection = (%s, %s, %s, %v)", gate, up, down, err)
	}

	if _, _, _, _, err := SelectDonorMLP(weights, 7); err == nil {
		t.Fatal("out-of-range triple index accepted")
	}
	incomplete := map[string][]float32{"blk.0.ffn_gate.weight": nil, "blk.0.ffn_up.weight": nil}
	if _, _, _, _, err := SelectDonorMLP(incomplete, 0); err == nil {
		t.Fatal("incomplete MLP triple accepted")
	}
}
