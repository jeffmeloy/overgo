package tensorcatalog

import (
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
)

// DenseLayerNames maps a dense decoder layer's Hugging Face tensor suffix to
// its GGUF suffix: the one naming contract the HF/GGUF converter writes with
// and the dense-causal loader reads back by.
var DenseLayerNames = map[string]string{
	"input_layernorm.weight":          "attn_norm.weight",
	"post_attention_layernorm.weight": "ffn_norm.weight",
	"self_attn.q_proj.weight":         "attn_q.weight",
	"self_attn.q_proj.bias":           "attn_q.bias",
	"self_attn.k_proj.weight":         "attn_k.weight",
	"self_attn.k_proj.bias":           "attn_k.bias",
	"self_attn.v_proj.weight":         "attn_v.weight",
	"self_attn.v_proj.bias":           "attn_v.bias",
	"self_attn.o_proj.weight":         "attn_output.weight",
	"self_attn.o_proj.bias":           "attn_output.bias",
	"mlp.gate_proj.weight":            "ffn_gate.weight",
	"mlp.gate_proj.bias":              "ffn_gate.bias",
	"mlp.up_proj.weight":              "ffn_up.weight",
	"mlp.up_proj.bias":                "ffn_up.bias",
	"mlp.down_proj.weight":            "ffn_down.weight",
	"mlp.down_proj.bias":              "ffn_down.bias",
}

// hfDenseLayerNames inverts DenseLayerNames once: the GGUF-to-HF direction
// the training loader consumes.
var hfDenseLayerNames = func() map[string]string {
	inverted := make(map[string]string, len(DenseLayerNames))
	for hf, gguf := range DenseLayerNames {
		inverted[gguf] = hf
	}
	return inverted
}()

// HFDenseTensorName maps one GGUF tensor name back to the HF catalog name
// the dense trainer indexes by. The mapping is the same table conversion
// writes with, so a round trip is exact; a GGUF name outside the dense
// vocabulary reports false rather than guessing.
func HFDenseTensorName(name string) (string, bool) {
	switch name {
	case "token_embd.weight":
		return "model.embed_tokens.weight", true
	case "output_norm.weight":
		return "model.norm.weight", true
	case "output.weight":
		return "lm_head.weight", true
	case "output.bias":
		return "lm_head.bias", true
	}
	rest, ok := strings.CutPrefix(name, "blk.")
	if !ok {
		return "", false
	}
	block, suffix, ok := strings.Cut(rest, ".")
	if !ok {
		return "", false
	}
	if _, err := strconv.ParseUint(block, binaryschema.DecimalRadix, binaryschema.Width32Bits); err != nil {
		return "", false
	}
	hfSuffix, ok := hfDenseLayerNames[suffix]
	if !ok {
		return "", false
	}
	return "model.layers." + block + "." + hfSuffix, true
}
