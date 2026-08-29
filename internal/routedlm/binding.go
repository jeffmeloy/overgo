package routedlm

import (
	"fmt"
	"slices"
	"strings"
)

// BranchBinding: checkpoint naming contract for a two-branch routed LM.
// Branch 0 is the base name; branch 1 appends Suffix at the owning module
// for every path in ForkPaths. QK-norm scales are section lists whose
// element counts sum to head_dim. Ported from adaptive_new go/extmodel
// tensorStackBranch / compileBranchedCausalGQAOperatorGroups.
type BranchBinding struct {
	LayerPrefix   string    // per-layer tensor prefix format, one %d (layer index)
	Suffix        string    // branch-1 name suffix, e.g. "_v" or "_mot_gen"
	ForkPaths     []string  // sorted ".weight"-trimmed layer paths that fork per branch
	QNormSections []string  // per-head Q-norm leaf suffixes, concatenated to head_dim
	KNormSections []string  // per-head K-norm leaf suffixes, concatenated to head_dim
	RopeThetaKeys []string  // per-QK-norm-section config key naming the rope base
	EmbedName     string    // token embedding tensor
	LMHeadName    string    // head tensor (falls back to tied embeddings)
	FinalNormName [2]string // per-branch final norm tensor (equal entries = shared)
	ConfigSection []string  // nested config.json path holding the LM fields; nil = flat
}

func (b BranchBinding) validate() error {
	switch {
	case strings.Count(b.LayerPrefix, "%d") != 1:
		return fmt.Errorf("routed lm binding: layer prefix %q needs exactly one %%d", b.LayerPrefix)
	case b.Suffix == "" || len(b.ForkPaths) == 0 || !slices.IsSorted(b.ForkPaths):
		return fmt.Errorf("routed lm binding: missing suffix or unsorted fork paths")
	case len(b.QNormSections) == 0 || len(b.KNormSections) != len(b.QNormSections):
		return fmt.Errorf("routed lm binding: missing or mismatched QK-norm sections")
	case len(b.RopeThetaKeys) != len(b.QNormSections):
		return fmt.Errorf("routed lm binding: rope theta keys %d != norm sections %d", len(b.RopeThetaKeys), len(b.QNormSections))
	case b.EmbedName == "" || b.FinalNormName[0] == "" || b.FinalNormName[1] == "":
		return fmt.Errorf("routed lm binding: missing embed or final norm names")
	}
	for index, path := range b.ForkPaths {
		if strings.TrimSpace(path) == "" || index > 0 && path == b.ForkPaths[index-1] {
			return fmt.Errorf("routed lm binding: invalid fork path %q", path)
		}
	}
	return nil
}

func (b BranchBinding) forks(path string) bool {
	_, found := slices.BinarySearch(b.ForkPaths, path)
	return found
}

// branchedTensorName: branch suffix at its owning module (ported verbatim
// from adaptive extmodel branchedTensorName).
func branchedTensorName(prefix, suffix, branch string) string {
	if rest, ok := strings.CutPrefix(suffix, "mlp."); ok {
		return prefix + "mlp" + branch + "." + rest
	}
	return prefix + strings.TrimSuffix(suffix, ".weight") + branch + ".weight"
}

// LayerTensorName: branch-resolved name of a layer-relative tensor (port of
// adaptive stackTensorName; branch 1 = the fork stack).
func (b BranchBinding) LayerTensorName(layer, branch int, suffix string) string {
	prefix := fmt.Sprintf(b.LayerPrefix, layer)
	if branch == 0 || !b.forks(strings.TrimSuffix(suffix, ".weight")) {
		return prefix + suffix
	}
	return branchedTensorName(prefix, suffix, b.Suffix)
}

// RxBrainBinding: Hy-Embodied-RxBrain's `_v` fork — single shared QK-norm
// section and shared final norm (the degenerate case), flat config.json.
func RxBrainBinding() BranchBinding {
	return BranchBinding{
		LayerPrefix: "model.language_model.model.layers.%d.",
		Suffix:      "_v",
		ForkPaths: []string{
			"input_layernorm",
			"mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
			"post_attention_layernorm",
			"self_attn.k_proj", "self_attn.o_proj", "self_attn.q_proj", "self_attn.v_proj",
		},
		QNormSections: []string{"self_attn.query_layernorm.weight"},
		KNormSections: []string{"self_attn.key_layernorm.weight"},
		RopeThetaKeys: []string{"rope_theta"},
		EmbedName:     "model.language_model.model.embed_tokens.weight",
		LMHeadName:    "model.language_model.lm_head.weight",
		FinalNormName: [2]string{
			"model.language_model.model.norm.weight",
			"model.language_model.model.norm.weight",
		},
	}
}

// SenseNovaBinding: SenseNova-U1 MoT's `_mot_gen` fork — two QK-norm
// sections per branch (norm + norm_hw), per-branch final norm, LM fields
// under config.json llm_config.
func SenseNovaBinding() BranchBinding {
	return BranchBinding{
		LayerPrefix: "language_model.model.layers.%d.",
		Suffix:      "_mot_gen",
		ForkPaths: []string{
			"input_layernorm",
			"mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
			"post_attention_layernorm",
			"self_attn.k_norm", "self_attn.k_norm_hw", "self_attn.k_proj",
			"self_attn.o_proj",
			"self_attn.q_norm", "self_attn.q_norm_hw", "self_attn.q_proj",
			"self_attn.v_proj",
		},
		QNormSections: []string{"self_attn.q_norm.weight", "self_attn.q_norm_hw.weight"},
		KNormSections: []string{"self_attn.k_norm.weight", "self_attn.k_norm_hw.weight"},
		RopeThetaKeys: []string{"rope_theta", "rope_theta_hw"},
		EmbedName:     "language_model.model.embed_tokens.weight",
		LMHeadName:    "language_model.lm_head.weight",
		FinalNormName: [2]string{
			"language_model.model.norm.weight",
			"language_model.model.norm_mot_gen.weight",
		},
		ConfigSection: []string{"llm_config"},
	}
}
