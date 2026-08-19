package rxbrain

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Inventory classifies the checkpoint's weight map into the components the
// vision-QA path consumes. The mixture-of-transformers layout carries three
// expert branches per decoder layer -- base text, vision (_v), generation
// (_g) -- plus the shared attention layernorms, the visual tower, and the
// flow-matching adapters the vision-QA scope loads nothing from.
type Inventory struct {
	// TensorShard maps every tensor name to its shard file.
	TensorShard map[string]string
	// Counts per component, derived, never asserted.
	TextBranch, VisionBranch, GenerationBranch, VisualTower, FlowAdapters, Shared int
}

// decoder tensor suffixes every layer must carry for a branch to be whole.
var branchSuffixes = []string{
	"input_layernorm%s.weight",
	"post_attention_layernorm%s.weight",
	"mlp%s.down_proj.weight",
	"mlp%s.gate_proj.weight",
	"mlp%s.up_proj.weight",
	"self_attn.q_proj%s.weight",
	"self_attn.k_proj%s.weight",
	"self_attn.v_proj%s.weight",
	"self_attn.o_proj%s.weight",
}

// LoadInventory reads the sharded index and validates completeness for the
// branches the vision-QA path executes: every decoder layer must carry its
// full base-text and vision branch plus the shared attention layernorms, and
// the visual tower and token embedding must exist. The generation branch and
// flow adapters are counted but not required -- absent scope, absent claims.
func LoadInventory(directory string, config Config) (Inventory, error) {
	data, err := os.ReadFile(filepath.Join(directory, "model.safetensors.index.json"))
	if err != nil {
		return Inventory{}, fmt.Errorf("rxbrain: read index: %w", err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return Inventory{}, fmt.Errorf("rxbrain: parse index: %w", err)
	}
	if len(index.WeightMap) == 0 {
		return Inventory{}, fmt.Errorf("rxbrain: index carries no weight map")
	}
	inventory := Inventory{TensorShard: index.WeightMap}
	for name := range index.WeightMap {
		switch {
		case strings.HasPrefix(name, "model.visual."):
			inventory.VisualTower++
		case strings.Contains(name, "_g.") || strings.HasPrefix(name, "llm2vae.") ||
			strings.HasPrefix(name, "vae2llm.") || strings.HasPrefix(name, "latent_pos_embed.") ||
			strings.HasPrefix(name, "time_embedder."):
			if strings.Contains(name, "_g.") {
				inventory.GenerationBranch++
			} else {
				inventory.FlowAdapters++
			}
		case strings.Contains(name, "_v."):
			inventory.VisionBranch++
		case strings.Contains(name, "layernorm") && strings.Contains(name, "self_attn"):
			inventory.Shared++
		default:
			inventory.TextBranch++
		}
	}
	layerPrefix := "model.language_model.model.layers."
	for layer := 0; layer < config.NumHiddenLayers; layer++ {
		for _, branch := range []string{"", "_v"} {
			for _, suffix := range branchSuffixes {
				name := fmt.Sprintf("%s%d.%s", layerPrefix, layer, fmt.Sprintf(suffix, branch))
				if _, ok := index.WeightMap[name]; !ok {
					return Inventory{}, fmt.Errorf("rxbrain: layer %d missing %s", layer, fmt.Sprintf(suffix, branch))
				}
			}
		}
		for _, shared := range []string{"self_attn.query_layernorm.weight", "self_attn.key_layernorm.weight"} {
			name := fmt.Sprintf("%s%d.%s", layerPrefix, layer, shared)
			if _, ok := index.WeightMap[name]; !ok {
				return Inventory{}, fmt.Errorf("rxbrain: layer %d missing shared %s", layer, shared)
			}
		}
	}
	if _, ok := index.WeightMap["model.language_model.model.embed_tokens.weight"]; !ok {
		return Inventory{}, fmt.Errorf("rxbrain: token embedding absent")
	}
	if inventory.VisualTower == 0 {
		return Inventory{}, fmt.Errorf("rxbrain: visual tower absent")
	}
	return inventory, nil
}
