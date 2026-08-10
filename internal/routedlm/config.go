// Package routedlm is the host reference path for a dense causal transformer
// whose per-token weights are branch-selected by a modality mask: every layer
// carries two complete weight sets (branch 0 = text, branch 1 = vision) over
// shared primitives (RMSNorm, GQA attention with per-head QK-norm and
// dynamic-NTK rope, SiLU MLP), and each token row routes through exactly one
// set. Activations follow the reference bf16-rounding discipline.
//
// Ported from adaptive_new go/extmodel merged_patch_vision.go (prompt layer
// stack, resident KV, decode row) and modality_transformer_config.go
// (config, terminal head, prompt rendering).
package routedlm

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

type Config struct {
	HiddenSize            int            `json:"hidden_size"`
	IntermediateSize      int            `json:"intermediate_size"`
	NumHiddenLayers       int            `json:"num_hidden_layers"`
	NumAttentionHeads     int            `json:"num_attention_heads"`
	NumKeyValueHeads      int            `json:"num_key_value_heads"`
	HeadDim               int            `json:"head_dim"`
	RopeTheta             float64        `json:"rope_theta"`
	VocabSize             int            `json:"vocab_size"`
	MaxPositionEmbeddings int            `json:"max_position_embeddings"`
	RMSNormEps            float64        `json:"rms_norm_eps"`
	HiddenAct             string         `json:"hidden_act"`
	UseQKNorm             bool           `json:"use_qk_norm"`
	RopeScaling           map[string]any `json:"rope_scaling"`
}

func LoadConfig(modelDir string) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, fmt.Errorf("routed lm config: %w", err)
	}
	return cfg, cfg.validate()
}

func (cfg Config) validate() error {
	switch {
	case cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.NumHiddenLayers <= 0:
		return fmt.Errorf("routed lm config: invalid dimensions hidden=%d intermediate=%d layers=%d", cfg.HiddenSize, cfg.IntermediateSize, cfg.NumHiddenLayers)
	case cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 || cfg.HeadDim <= 0:
		return fmt.Errorf("routed lm config: invalid attention dimensions heads=%d kv_heads=%d head_dim=%d", cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim)
	case cfg.HiddenSize != cfg.NumAttentionHeads*cfg.HeadDim:
		return fmt.Errorf("routed lm config: hidden_size=%d differs from heads*head_dim=%d", cfg.HiddenSize, cfg.NumAttentionHeads*cfg.HeadDim)
	case cfg.VocabSize <= 0 || cfg.MaxPositionEmbeddings <= 0:
		return fmt.Errorf("routed lm config: invalid vocab/context vocab=%d max_positions=%d", cfg.VocabSize, cfg.MaxPositionEmbeddings)
	case cfg.RMSNormEps <= 0:
		return fmt.Errorf("routed lm config: rms_norm_eps=%g", cfg.RMSNormEps)
	case cfg.HiddenAct != "" && cfg.HiddenAct != "silu":
		return fmt.Errorf("routed lm config: unsupported hidden_act %q", cfg.HiddenAct)
	}
	return nil
}

func float64ConfigValue(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

// RopeInvFreqBase: the effective rotary theta. The declared dynamic-NTK
// rope_scaling only rescales theta by alpha^(d/(d-2)); factor/mscale are 1.0
// and beta_fast/beta_slow are unused by the reference implementation.
func RopeInvFreqBase(cfg Config) (float64, error) {
	if cfg.HeadDim <= 2 || cfg.RopeTheta <= 0 {
		return 0, fmt.Errorf("routed lm rope: invalid head_dim=%d rope_theta=%g", cfg.HeadDim, cfg.RopeTheta)
	}
	base := cfg.RopeTheta
	if cfg.RopeScaling != nil {
		if typ, _ := cfg.RopeScaling["type"].(string); typ == "dynamic" {
			if alpha, ok := float64ConfigValue(cfg.RopeScaling["alpha"]); ok && alpha != 0 {
				base = cfg.RopeTheta * math.Pow(alpha, float64(cfg.HeadDim)/float64(cfg.HeadDim-2))
			}
		}
	}
	return base, nil
}
