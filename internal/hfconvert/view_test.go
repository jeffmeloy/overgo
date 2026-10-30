package hfconvert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

// TestAQwen35ViewIsTheBytesTheConverterWrites holds the view of a Qwen 3.5 directory to the file Convert writes from
// it: its image, header, tensors and padding, is the converted file byte for byte, read whole and read in pieces that
// cross every boundary, so a model served from its own files reads what its conversion would have written. The
// directory carries the parts the converter transforms (the norms it shifts by one, A_log, the value heads it
// reorders, the convolution it squeezes) beside tensors it keeps as they are; the view builds the first on read and
// reads the second in place.
func TestAQwen35ViewIsTheBytesTheConverterWrites(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	text := map[string]any{
		"model_type": "qwen3_5_text", "max_position_embeddings": 16, "hidden_size": 8, "num_hidden_layers": 4,
		"mtp_num_hidden_layers": 1, "intermediate_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1,
		"head_dim": 4, "rms_norm_eps": 1e-6, "vocab_size": 32, "eos_token_id": 1, "full_attention_interval": 4,
		"linear_conv_kernel_dim": 3, "linear_key_head_dim": 2, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
		"linear_value_head_dim": 2,
		"layer_types":           []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		"rope_parameters":       map[string]any{"rope_theta": 10000, "partial_rotary_factor": 1, "mrope_section": []int{1, 1, 0}},
	}
	config, err := json.Marshal(map[string]any{"model_type": "qwen3_5", "text_config": text})
	if err != nil {
		t.Fatal(err)
	}
	vocabulary := map[string]int{}
	for id := range 32 {
		vocabulary[fmt.Sprintf("t%d", id)] = id
	}
	tokenizer, err := json.Marshal(map[string]any{"model": map[string]any{"vocab": vocabulary, "merges": []string{}}})
	if err != nil {
		t.Fatal(err)
	}
	testutil.WriteTextFile(t, directory, "config.json", string(config))
	testutil.WriteTextFile(t, directory, "tokenizer.json", string(tokenizer))
	shapes := map[string][]int{
		"model.language_model.embed_tokens.weight": {32, 8},
		"model.language_model.norm.weight":         {8},
		"mtp.fc.weight":                            {8, 16},
		"mtp.pre_fc_norm_embedding.weight":         {8},
		"mtp.pre_fc_norm_hidden.weight":            {8},
		"mtp.norm.weight":                          {8},
	}
	for block := range 5 {
		prefix := fmt.Sprintf("model.language_model.layers.%d.", block)
		if block == 4 {
			prefix = "mtp.layers.0."
		}
		for name, shape := range map[string][]int{
			"input_layernorm.weight": {8}, "post_attention_layernorm.weight": {8},
			"mlp.gate_proj.weight": {16, 8}, "mlp.up_proj.weight": {16, 8}, "mlp.down_proj.weight": {8, 16},
		} {
			shapes[prefix+name] = shape
		}
		attention := map[string][]int{
			"self_attn.q_proj.weight": {16, 8}, "self_attn.k_proj.weight": {4, 8}, "self_attn.v_proj.weight": {4, 8},
			"self_attn.o_proj.weight": {8, 8}, "self_attn.q_norm.weight": {4}, "self_attn.k_norm.weight": {4},
		}
		if block < 3 {
			attention = map[string][]int{
				"linear_attn.in_proj_qkv.weight": {16, 8}, "linear_attn.in_proj_z.weight": {8, 8},
				"linear_attn.in_proj_a.weight": {4, 8}, "linear_attn.in_proj_b.weight": {4, 8}, "linear_attn.A_log": {4},
				"linear_attn.dt_bias": {4}, "linear_attn.conv1d.weight": {16, 1, 3}, "linear_attn.norm.weight": {2},
				"linear_attn.out_proj.weight": {8, 8},
			}
		}
		for name, shape := range attention {
			shapes[prefix+name] = shape
		}
	}
	weights := map[string][]float32{}
	seed := 0
	for name, shape := range shapes {
		elements := 1
		for _, dimension := range shape {
			elements *= dimension
		}
		values := make([]float32, elements)
		for index := range values {
			seed++
			values[index] = float32(math.Sin(float64(seed)) / 4)
		}
		weights[name] = values
	}
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	converted := filepath.Join(t.TempDir(), "converted.gguf")
	if _, err := Convert(Options{Directory: directory, OutputPath: converted, Name: "fixture"}); err != nil {
		t.Fatal(err)
	}
	written, err := os.ReadFile(converted)
	if err != nil {
		t.Fatal(err)
	}
	file, err := View(directory, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	shards := file.Shards()
	if len(shards) != 1 || shards[0].Size != uint64(len(written)) {
		t.Fatalf("the view's image is %+v, want one of the converted file's %d bytes", shards, len(written))
	}
	whole := make([]byte, len(written))
	if _, err := shards[0].Source.ReadAt(whole, 0); err != nil {
		t.Fatal(err)
	}
	if at := firstDifference(whole, written); at >= 0 {
		t.Fatalf("the view read whole differs from the converted file at byte %d of %d", at, len(written))
	}
	pieces := make([]byte, 0, len(written))
	for offset := 0; offset < len(written); offset += 7 {
		piece := make([]byte, min(7, len(written)-offset))
		if _, err := shards[0].Source.ReadAt(piece, int64(offset)); err != nil {
			t.Fatal(err)
		}
		pieces = append(pieces, piece...)
	}
	if at := firstDifference(pieces, written); at >= 0 {
		t.Fatalf("the view read in pieces differs from the converted file at byte %d", at)
	}
	opened, err := gguf.Open(converted)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if len(file.Tensors) != len(opened.Tensors) || len(file.Metadata) != len(opened.Metadata) {
		t.Errorf("the view parses %d tensors and %d metadata, the converted file %d and %d", len(file.Tensors), len(file.Metadata), len(opened.Tensors), len(opened.Metadata))
	}
}

// firstDifference is the first index where two byte slices differ, or -1.
func firstDifference(left, right []byte) int {
	if bytes.Equal(left, right) {
		return -1
	}
	for index := range min(len(left), len(right)) {
		if left[index] != right[index] {
			return index
		}
	}
	return min(len(left), len(right))
}
