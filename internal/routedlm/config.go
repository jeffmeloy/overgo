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
	ropeBases             map[string]float64
}

// RopeBase: numeric config field by the key a binding names (same config
// section the LM fields were parsed from).
func (cfg Config) RopeBase(key string) (float64, bool) {
	value, ok := cfg.ropeBases[key]
	return value, ok
}

// LoadConfig: LM fields from config.json, descending the binding's config
// section path (nil = flat top-level fields).
func LoadConfig(modelDir string, b BranchBinding) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return Config{}, err
	}
	return parseConfig(raw, b.ConfigSection)
}

func parseConfig(raw []byte, sectionPath []string) (Config, error) {
	for _, section := range sectionPath {
		var node map[string]json.RawMessage
		if err := json.Unmarshal(raw, &node); err != nil {
			return Config{}, fmt.Errorf("routed lm config: %w", err)
		}
		sub, ok := node[section]
		if !ok {
			return Config{}, fmt.Errorf("routed lm config: missing section %s", section)
		}
		raw = sub
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, fmt.Errorf("routed lm config: %w", err)
	}
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil {
		return Config{}, fmt.Errorf("routed lm config: %w", err)
	}
	cfg.ropeBases = make(map[string]float64)
	for key, value := range fields {
		if number, ok := float64ConfigValue(value); ok {
			cfg.ropeBases[key] = number
		}
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
	return ropeSectionBase(cfg.RopeTheta, cfg.HeadDim, cfg.RopeScaling)
}

// ropeSectionBase: dynamic-NTK-adjusted rope base for one rotary section of
// the given width (width = head_dim is the full-width case above).
func ropeSectionBase(theta float64, width int, scaling map[string]any) (float64, error) {
	if width <= 2 || theta <= 0 {
		return 0, fmt.Errorf("routed lm rope: invalid section width=%d theta=%g", width, theta)
	}
	base := theta
	if scaling != nil {
		if typ, _ := scaling["type"].(string); typ == "dynamic" {
			if alpha, ok := float64ConfigValue(scaling["alpha"]); ok && alpha != 0 {
				base = theta * math.Pow(alpha, float64(width)/float64(width-2))
			}
		}
	}
	return base, nil
}
