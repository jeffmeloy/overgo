package testutil

import (
	"fmt"
	"testing"

	"overgo/internal/gguf"
)

// The miniature Gemma 4 E4B the shared-KV tests decode: every dimension is
// small enough to build on the host and large enough to cross the cache's page
// bands.
const (
	// g4Hidden is the model width.
	g4Hidden = uint64(16)
	// g4Vocabulary is the token count: the specials and twenty letters.
	g4Vocabulary = uint64(23)
	// g4PerLayer is the width of each layer's input adapter.
	g4PerLayer = uint64(8)
	// g4Heads is the query head count.
	g4Heads = uint64(4)
	// g4KVHeads is the key and value head count.
	g4KVHeads = uint64(2)
	// g4SlidingBase is the head width of the sliding layers.
	g4SlidingBase = uint64(8)
	// g4FullBase is the head width of the full layers.
	g4FullBase = uint64(16)
	// g4Inner is the feed-forward width.
	g4Inner = uint64(32)
	// g4Window is the sliding layers' attention window in tokens.
	g4Window = uint32(3)
	// g4Layers is the layer count.
	g4Layers = 4
	// g4Shared is how many of the last layers attend the keys and values of
	// earlier ones instead of projecting their own.
	g4Shared = 2
	// g4ControlTokenType is the GGUF token type of begin and end of sequence.
	g4ControlTokenType = 3
	// g4NormalTokenType is the GGUF token type of the letters.
	g4NormalTokenType = 1
	// g4UnknownTokenType is the GGUF token type of the unknown token.
	g4UnknownTokenType = 2
	// g4FullRopeBase is the rotary base of the full layers.
	g4FullRopeBase = 1_000_000
	// g4SlidingRopeBase is the rotary base of the sliding layers.
	g4SlidingRopeBase = 10_000
	// g4NormEpsilon is the RMS norm epsilon.
	g4NormEpsilon = 1e-6
	// g4LogitSoftcap is the cap the final logits are squashed to.
	g4LogitSoftcap = 3
	// g4HalfWidth halves a width to its rotary frequency count.
	g4HalfWidth = 2
	// g4EndOfSequenceToken is the id of the end-of-sequence token.
	g4EndOfSequenceToken = 2
)

// HermeticGemma4SharedKVGGUF writes the Gemma 4 E4B layer kinds in F32
// miniature: a sliding layer and a full layer that project their own keys and
// values, then a sliding and a full layer that attend those, with a per-layer
// input adapter on every layer, as the real model's last 18 layers attend the
// last two that project.
func HermeticGemma4SharedKVGGUF(t testing.TB, context uint32) string {
	t.Helper()
	sliding := []bool{true, false, true, false}
	seed := 0
	var tensors []gguf.TensorData
	add := func(name string, shape ...uint64) {
		seed++
		tensors = append(tensors, GGUFTensorF32(name, shape, seed))
	}
	perLayerWidth := g4Layers * g4PerLayer
	add("token_embd.weight", g4Hidden, g4Vocabulary)
	add("per_layer_token_embd.weight", perLayerWidth, g4Vocabulary)
	add("per_layer_model_proj.weight", g4Hidden, perLayerWidth)
	add("per_layer_proj_norm.weight", g4PerLayer)
	add("rope_freqs.weight", g4FullBase/g4HalfWidth)
	for layer := range g4Layers {
		name := func(role string) string { return fmt.Sprintf("blk.%d.%s.weight", layer, role) }
		width := g4FullBase
		if sliding[layer] {
			width = g4SlidingBase
		}
		query, key := g4Heads*width, g4KVHeads*width
		add(name("attn_norm"), g4Hidden)
		add(name("attn_q"), g4Hidden, query)
		add(name("attn_q_norm"), width)
		add(name("attn_k_norm"), width)
		if layer < g4Layers-g4Shared {
			add(name("attn_k"), g4Hidden, key)
			add(name("attn_v"), g4Hidden, key)
		}
		add(name("attn_output"), query, g4Hidden)
		add(name("post_attention_norm"), g4Hidden)
		add(name("ffn_norm"), g4Hidden)
		add(name("ffn_gate"), g4Hidden, g4Inner)
		add(name("ffn_up"), g4Hidden, g4Inner)
		add(name("ffn_down"), g4Inner, g4Hidden)
		add(name("post_ffw_norm"), g4Hidden)
		add(name("per_layer_inp_gate"), g4Hidden, g4PerLayer)
		add(name("per_layer_proj"), g4PerLayer, g4Hidden)
		add(name("per_layer_post_norm"), g4Hidden)
		if layer < g4Layers-1 {
			add(name("layer_output_scale"), 1)
		}
	}
	add("output_norm.weight", g4Hidden)
	layerInts := func(value int32) []int32 {
		values := make([]int32, g4Layers)
		for index := range values {
			values[index] = value
		}
		return values
	}
	tokens := []string{"<unk>", "<s>", "</s>"}
	for letter := range int(g4Vocabulary) - len(tokens) {
		tokens = append(tokens, string(rune('a'+letter)))
	}
	tokenTypes := make([]int32, g4Vocabulary)
	for index := range tokenTypes {
		tokenTypes[index] = g4NormalTokenType
	}
	copy(tokenTypes, []int32{g4UnknownTokenType, g4ControlTokenType, g4ControlTokenType})
	metadata := []gguf.Metadata{
		GGUFScalar("general.architecture", gguf.ValueTypeString, "gemma4"),
		GGUFScalar("general.name", gguf.ValueTypeString, "hermetic-gemma4-shared-kv"),
		GGUFScalar("gemma4.block_count", gguf.ValueTypeUint32, uint32(g4Layers)),
		GGUFScalar("gemma4.context_length", gguf.ValueTypeUint32, context),
		GGUFScalar("gemma4.embedding_length", gguf.ValueTypeUint32, uint32(g4Hidden)),
		GGUFArray("gemma4.feed_forward_length", gguf.ValueTypeInt32, layerInts(int32(g4Inner))),
		GGUFScalar("gemma4.attention.head_count", gguf.ValueTypeUint32, uint32(g4Heads)),
		GGUFArray("gemma4.attention.head_count_kv", gguf.ValueTypeInt32, layerInts(int32(g4KVHeads))),
		GGUFScalar("gemma4.attention.key_length", gguf.ValueTypeUint32, uint32(g4FullBase)),
		GGUFScalar("gemma4.attention.value_length", gguf.ValueTypeUint32, uint32(g4FullBase)),
		GGUFScalar("gemma4.attention.key_length_swa", gguf.ValueTypeUint32, uint32(g4SlidingBase)),
		GGUFScalar("gemma4.attention.value_length_swa", gguf.ValueTypeUint32, uint32(g4SlidingBase)),
		GGUFScalar("gemma4.rope.dimension_count", gguf.ValueTypeUint32, uint32(g4FullBase)),
		GGUFScalar("gemma4.rope.dimension_count_swa", gguf.ValueTypeUint32, uint32(g4SlidingBase)),
		GGUFScalar("gemma4.rope.freq_base", gguf.ValueTypeFloat32, float32(g4FullRopeBase)),
		GGUFScalar("gemma4.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(g4SlidingRopeBase)),
		GGUFScalar("gemma4.attention.sliding_window", gguf.ValueTypeUint32, g4Window),
		GGUFArray("gemma4.attention.sliding_window_pattern", gguf.ValueTypeBool, sliding),
		GGUFScalar("gemma4.attention.shared_kv_layers", gguf.ValueTypeUint32, uint32(g4Shared)),
		GGUFScalar("gemma4.embedding_length_per_layer_input", gguf.ValueTypeUint32, uint32(g4PerLayer)),
		GGUFScalar("gemma4.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(g4NormEpsilon)),
		GGUFScalar("gemma4.final_logit_softcapping", gguf.ValueTypeFloat32, float32(g4LogitSoftcap)),
		GGUFScalar("gemma4.vocab_size", gguf.ValueTypeUint32, uint32(g4Vocabulary)),
		GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, g4Vocabulary)),
		GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32, tokenTypes),
		GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(g4EndOfSequenceToken)),
	}
	return TempGGUF(t, "hermetic-gemma4-shared-kv.gguf", metadata, tensors)
}
