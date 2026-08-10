package routedlm

import "testing"

// Table values verified against the checkpoints' model.safetensors.index.json.
func TestBindingLayerTensorNames(t *testing.T) {
	rx, sn := RxBrainBinding(), SenseNovaBinding()
	if err := rx.validate(); err != nil {
		t.Fatal(err)
	}
	if err := sn.validate(); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		b              BranchBinding
		layer, branch  int
		suffix, expect string
	}{
		// RxBrain `_v`: leaf placement.
		{rx, 0, 0, "self_attn.q_proj.weight", "model.language_model.model.layers.0.self_attn.q_proj.weight"},
		{rx, 0, 1, "self_attn.q_proj.weight", "model.language_model.model.layers.0.self_attn.q_proj_v.weight"},
		{rx, 5, 1, "input_layernorm.weight", "model.language_model.model.layers.5.input_layernorm_v.weight"},
		{rx, 5, 1, "post_attention_layernorm.weight", "model.language_model.model.layers.5.post_attention_layernorm_v.weight"},
		// RxBrain `_v`: mlp module placement.
		{rx, 7, 1, "mlp.gate_proj.weight", "model.language_model.model.layers.7.mlp_v.gate_proj.weight"},
		{rx, 7, 0, "mlp.gate_proj.weight", "model.language_model.model.layers.7.mlp.gate_proj.weight"},
		// RxBrain shared QK norms: not forked, both branches identical.
		{rx, 3, 0, "self_attn.query_layernorm.weight", "model.language_model.model.layers.3.self_attn.query_layernorm.weight"},
		{rx, 3, 1, "self_attn.query_layernorm.weight", "model.language_model.model.layers.3.self_attn.query_layernorm.weight"},
		{rx, 3, 1, "self_attn.key_layernorm.weight", "model.language_model.model.layers.3.self_attn.key_layernorm.weight"},
		// SenseNova `_mot_gen`: leaf placement.
		{sn, 0, 0, "self_attn.q_proj.weight", "language_model.model.layers.0.self_attn.q_proj.weight"},
		{sn, 0, 1, "self_attn.q_proj.weight", "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"},
		{sn, 2, 1, "self_attn.o_proj.weight", "language_model.model.layers.2.self_attn.o_proj_mot_gen.weight"},
		{sn, 2, 1, "input_layernorm.weight", "language_model.model.layers.2.input_layernorm_mot_gen.weight"},
		// SenseNova `_mot_gen`: mlp module placement.
		{sn, 41, 1, "mlp.gate_proj.weight", "language_model.model.layers.41.mlp_mot_gen.gate_proj.weight"},
		{sn, 41, 1, "mlp.down_proj.weight", "language_model.model.layers.41.mlp_mot_gen.down_proj.weight"},
		{sn, 41, 0, "mlp.up_proj.weight", "language_model.model.layers.41.mlp.up_proj.weight"},
		// SenseNova forked two-section QK norms.
		{sn, 9, 0, "self_attn.q_norm.weight", "language_model.model.layers.9.self_attn.q_norm.weight"},
		{sn, 9, 0, "self_attn.q_norm_hw.weight", "language_model.model.layers.9.self_attn.q_norm_hw.weight"},
		{sn, 9, 1, "self_attn.q_norm.weight", "language_model.model.layers.9.self_attn.q_norm_mot_gen.weight"},
		{sn, 9, 1, "self_attn.q_norm_hw.weight", "language_model.model.layers.9.self_attn.q_norm_hw_mot_gen.weight"},
		{sn, 9, 1, "self_attn.k_norm.weight", "language_model.model.layers.9.self_attn.k_norm_mot_gen.weight"},
		{sn, 9, 1, "self_attn.k_norm_hw.weight", "language_model.model.layers.9.self_attn.k_norm_hw_mot_gen.weight"},
	}
	for _, c := range cases {
		if got := c.b.LayerTensorName(c.layer, c.branch, c.suffix); got != c.expect {
			t.Errorf("layer=%d branch=%d %s: got %s, want %s", c.layer, c.branch, c.suffix, got, c.expect)
		}
	}
}

func TestBindingTerminalNames(t *testing.T) {
	rx, sn := RxBrainBinding(), SenseNovaBinding()
	if rx.FinalNormName[0] != rx.FinalNormName[1] || rx.FinalNormName[0] != "model.language_model.model.norm.weight" {
		t.Fatalf("rxbrain final norms %v", rx.FinalNormName)
	}
	if sn.FinalNormName[0] != "language_model.model.norm.weight" || sn.FinalNormName[1] != "language_model.model.norm_mot_gen.weight" {
		t.Fatalf("sensenova final norms %v", sn.FinalNormName)
	}
	if sn.EmbedName != "language_model.model.embed_tokens.weight" || sn.LMHeadName != "language_model.lm_head.weight" {
		t.Fatalf("sensenova terminal names %s %s", sn.EmbedName, sn.LMHeadName)
	}
}

func TestParseConfigSectionPath(t *testing.T) {
	flat := []byte(`{"hidden_size":5120,"intermediate_size":25600,"num_hidden_layers":32,
		"num_attention_heads":40,"num_key_value_heads":8,"head_dim":128,"rope_theta":10000,
		"vocab_size":128,"max_position_embeddings":4096,"rms_norm_eps":1e-6,"hidden_act":"silu"}`)
	cfg, err := parseConfig(flat, nil)
	if err != nil || cfg.HiddenSize != 5120 {
		t.Fatalf("flat parse hidden=%d err=%v", cfg.HiddenSize, err)
	}
	nested := append([]byte(`{"model_type":"mot","llm_config":`), flat...)
	nested = append(nested, '}')
	cfg, err = parseConfig(nested, []string{"llm_config"})
	if err != nil || cfg.HiddenSize != 5120 || cfg.NumHiddenLayers != 32 {
		t.Fatalf("nested parse hidden=%d layers=%d err=%v", cfg.HiddenSize, cfg.NumHiddenLayers, err)
	}
	if _, err = parseConfig(nested, []string{"missing_section"}); err == nil {
		t.Fatal("missing section accepted")
	}
	// Flat read of a nested config must fail validation, not silently zero.
	if _, err = parseConfig(nested, nil); err == nil {
		t.Fatal("nested config accepted as flat")
	}
}
