package hfconvert

import (
	"bytes"
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
// directory carries the parts the converter transforms beside tensors it keeps as they are; the view builds the first
// on read and reads the second in place. gguf.Open serves the directory as the same view.
func TestAQwen35ViewIsTheBytesTheConverterWrites(t *testing.T) {
	t.Parallel()
	requireViewIsConversion(t, testutil.Qwen35Directory(t))
}

// TestAProfileViewIsTheBytesTheConverterWrites holds the view of a directory of each architecture the converter writes
// by profile to the file Convert writes from it, as a Qwen 3.5 directory's is: a qwen2 checkpoint, whose attention
// biases the conversion promotes to F32, and a llama one, whose q and k rows it permutes from rotate-half, both built
// on read beside the tensors read in place. Qwen2.5-0.5B is a qwen2 checkpoint, Carbon-500M and MiniCPM5-1B llama ones,
// so each is served from its own files reading what its conversion holds.
func TestAProfileViewIsTheBytesTheConverterWrites(t *testing.T) {
	t.Parallel()
	for _, modelType := range []string{"qwen2", "llama"} {
		t.Run(modelType, func(t *testing.T) {
			t.Parallel()
			directory := profileDirectory(t, modelType)
			if !Viewed(directory) {
				t.Fatalf("a %s directory is not served as it is", modelType)
			}
			requireViewIsConversion(t, directory)
		})
	}
}

// profileWeightScale keeps the fixture's weights within a quarter of zero, the size of an initialized model's.
const profileWeightScale = 0.25

// profileDirectory writes a Hugging Face directory of modelType at the smallest shapes a profile conversion takes: one
// layer of two heads of two and one key-value head, a four-token vocabulary, each weight value distinct, and for qwen2
// the attention biases.
func profileDirectory(t *testing.T, modelType string) string {
	t.Helper()
	directory := t.TempDir()
	testutil.WriteTextFile(t, directory, "config.json", `{"model_type":"`+modelType+`","hidden_size":4,"num_hidden_layers":1,`+
		`"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":4,"max_position_embeddings":16,`+
		`"rms_norm_eps":1e-6,"rope_theta":10000,"tie_word_embeddings":false}`)
	testutil.WriteTextFile(t, directory, "tokenizer.json", `{"model":{"vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}`)
	shapes := map[string][]int{
		"model.embed_tokens.weight": {4, 4}, "model.norm.weight": {4}, "lm_head.weight": {4, 4},
		"model.layers.0.self_attn.q_proj.weight": {4, 4}, "model.layers.0.self_attn.k_proj.weight": {2, 4},
		"model.layers.0.self_attn.v_proj.weight": {2, 4}, "model.layers.0.self_attn.o_proj.weight": {4, 4},
		"model.layers.0.mlp.gate_proj.weight": {8, 4}, "model.layers.0.mlp.up_proj.weight": {8, 4},
		"model.layers.0.mlp.down_proj.weight": {4, 8}, "model.layers.0.input_layernorm.weight": {4},
		"model.layers.0.post_attention_layernorm.weight": {4},
	}
	if archProfiles[modelType].attentionBias {
		shapes["model.layers.0.self_attn.q_proj.bias"] = []int{4}
		shapes["model.layers.0.self_attn.k_proj.bias"] = []int{2}
		shapes["model.layers.0.self_attn.v_proj.bias"] = []int{2}
	}
	weights, seed := map[string][]float32{}, 0
	for name, shape := range shapes {
		elements := 1
		for _, dimension := range shape {
			elements *= dimension
		}
		values := make([]float32, elements)
		for index := range values {
			seed++
			values[index] = float32(math.Sin(float64(seed)) * profileWeightScale)
		}
		weights[name] = values
	}
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	return directory
}

// requireViewIsConversion holds the view of directory to the file Convert writes from it, byte for byte, read whole
// and read in pieces that cross every boundary, and gguf.Open to serving the directory as the same view.
func requireViewIsConversion(t *testing.T, directory string) {
	t.Helper()
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
	opened, err := gguf.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if opened.ViewDirectory() == "" || len(opened.Tensors) != len(file.Tensors) || len(opened.Metadata) != len(file.Metadata) {
		t.Errorf("gguf.Open of the directory = %d tensors, %d metadata, viewing %q; want the view of it", len(opened.Tensors), len(opened.Metadata), opened.ViewDirectory())
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
