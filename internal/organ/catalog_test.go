package organ

import "testing"

// TestOrganCatalogIndexesComponents pins the compiled catalog as the
// component authority: classification happens once at compile time, complete
// gated-MLP triples index deterministically across HF and GGUF naming,
// incomplete groups are absent rather than half-bound, ambiguous bindings are
// refused by name, and lookups outside the index report absent.
func TestOrganCatalogIndexesComponents(t *testing.T) {
	catalog, err := CompileCatalog([]string{
		"model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight",
		"model.layers.1.mlp.gate_proj.weight", "model.layers.1.mlp.up_proj.weight", "model.layers.1.mlp.down_proj.weight",
		"model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight", "model.norm.weight",
	})
	if err != nil {
		t.Fatal(err)
	}
	if catalog.MLPCount() != 2 {
		t.Fatalf("MLPCount = %d, want 2", catalog.MLPCount())
	}
	component, ok := catalog.MLP(1)
	if !ok || component.Ordinal != 1 || component.Gate != "model.layers.1.mlp.gate_proj.weight" ||
		component.Up != "model.layers.1.mlp.up_proj.weight" || component.Down != "model.layers.1.mlp.down_proj.weight" ||
		component.Contract.Role != RoleMLPGate {
		t.Fatalf("indexed component = (%+v, %v)", component, ok)
	}

	gguf, err := CompileCatalog([]string{
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.attn_q.weight", "token_embd.weight",
	})
	if err != nil {
		t.Fatal(err)
	}
	component, ok = gguf.MLP(0)
	if !ok || component.Gate != "blk.0.ffn_gate.weight" || component.Down != "blk.0.ffn_down.weight" {
		t.Fatalf("GGUF component = (%+v, %v)", component, ok)
	}

	incomplete, err := CompileCatalog([]string{"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"})
	if err != nil {
		t.Fatal(err)
	}
	if incomplete.MLPCount() != 0 {
		t.Fatal("incomplete triple compiled as a component")
	}
	if _, ok := incomplete.MLP(0); ok {
		t.Fatal("lookup outside the index reported present")
	}
	if _, err := CompileCatalog([]string{
		"model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.bias",
	}); err == nil {
		t.Fatal("ambiguous duplicate role binding accepted")
	}
}
