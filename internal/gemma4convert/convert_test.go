package gemma4convert

import (
	"encoding/binary"
	"io"
	"math"
	"testing"

	"llamacpp2go/internal/gguf"
)

func TestModelTensorName(t *testing.T) {
	for source, want := range map[string]string{
		"model.language_model.embed_tokens.weight":                         "token_embd.weight",
		"model.language_model.norm.weight":                                 "output_norm.weight",
		"model.language_model.layers.17.input_layernorm.weight":            "blk.17.attn_norm.weight",
		"model.language_model.layers.17.self_attn.q_proj.weight":           "blk.17.attn_q.weight",
		"model.language_model.layers.17.mlp.down_proj.weight":              "blk.17.ffn_down.weight",
		"model.language_model.layers.17.post_feedforward_layernorm.weight": "blk.17.post_ffw_norm.weight",
		"model.language_model.layers.17.layer_scalar":                      "blk.17.layer_output_scale.weight",
	} {
		got, ok := modelTensorName(source)
		if !ok || got != want {
			t.Fatalf("mapping %q = %q, %t; want %q", source, got, ok, want)
		}
	}
	if _, ok := modelTensorName("model.embed_vision.patch_dense.weight"); ok {
		t.Fatal("projector tensor mapped into language model")
	}
}

func TestReverseShape(t *testing.T) {
	actual := reverseShape([]uint64{2, 3, 5})
	want := []uint64{5, 3, 2}
	for index := range want {
		if actual[index] != want[index] {
			t.Fatalf("shape = %v, want %v", actual, want)
		}
	}
}

func TestProportionalRopeTensor(t *testing.T) {
	var config modelConfig
	config.Text.GlobalHeadDim = 16
	full := config.Text.Rope["full_attention"]
	full.PartialRotary = 0.25
	if config.Text.Rope == nil {
		config.Text.Rope = make(map[string]struct {
			PartialRotary float32 `json:"partial_rotary_factor"`
			Theta         float32 `json:"rope_theta"`
			Type          string  `json:"rope_type"`
		})
	}
	config.Text.Rope["full_attention"] = full
	tensor := proportionalRopeTensor(config)
	if tensor.Name != "rope_freqs.weight" || tensor.Type != gguf.DTypeF32 ||
		len(tensor.Shape) != 1 || tensor.Shape[0] != 8 {
		t.Fatalf("tensor = %+v", tensor)
	}
	encoded, err := io.ReadAll(tensor.Data)
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 8; index++ {
		got := math.Float32frombits(binary.LittleEndian.Uint32(encoded[index*4:]))
		want := float32(1e30)
		if index < 2 {
			want = 1
		}
		if got != want {
			t.Fatalf("factor %d = %g, want %g", index, got, want)
		}
	}
}

func TestProjectorMetadataMatchesPinnedLoader(t *testing.T) {
	var config modelConfig
	config.Text.HiddenSize = 3840
	config.Vision.Embedding = 3840
	config.Vision.PatchSize = 16
	config.Vision.PoolingSize = 3
	config.Vision.RMSEpsilon = 1e-6
	config.Audio.Embedding = 640
	config.Audio.RMSEpsilon = 1e-6
	metadata := projectorMetadata("fixture", config)
	byKey := make(map[string]gguf.Value, len(metadata))
	for _, item := range metadata {
		byKey[item.Key] = item.Value
	}
	for _, key := range []string{
		"clip.vision.image_size", "clip.vision.embedding_length",
		"clip.vision.feed_forward_length", "clip.vision.block_count",
		"clip.vision.attention.head_count", "clip.vision.image_mean", "clip.vision.image_std",
		"clip.audio.feed_forward_length", "clip.audio.block_count",
		"clip.audio.attention.head_count", "clip.audio.num_mel_bins",
	} {
		if _, ok := byKey[key]; !ok {
			t.Fatalf("metadata %q is missing", key)
		}
	}
}
