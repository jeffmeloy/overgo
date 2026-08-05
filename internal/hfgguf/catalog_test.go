package hfgguf

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"testing"

	"llamacpp2go/internal/hfrepo"
	"llamacpp2go/internal/safetensors"
)

func TestValidateDenseRepositoryUsesRuntimeCatalog(t *testing.T) {
	repository := denseFixture(t, "qwen2", true)
	spec, err := ValidateDenseRepository(repository)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen2" || spec.BlockCount != 1 || spec.VocabularySize != 32 || spec.KeyLength != 4 {
		t.Fatalf("spec = %+v", spec)
	}
}

func TestValidateDenseRepositoryRejectsUnmappedTensor(t *testing.T) {
	repository := denseFixture(t, "llama", false)
	repository.Tensors.Tensors["model.layers.0.unknown.weight"] = testTensor(t, "unknown", []uint64{8, 8})
	if _, err := ValidateDenseRepository(repository); err == nil {
		t.Fatal("unmapped tensor accepted")
	}
}

func TestDenseTensorDataReversesShapesAndPromotesVectors(t *testing.T) {
	repository := denseFixture(t, "qwen2", true)
	tensors, err := DenseTensorData(repository)
	if err != nil {
		t.Fatal(err)
	}
	for _, tensor := range tensors {
		switch tensor.Name {
		case "token_embd.weight":
			if len(tensor.Shape) != 2 || tensor.Shape[0] != 8 || tensor.Shape[1] != 32 {
				t.Fatalf("embedding shape = %v", tensor.Shape)
			}
		case "output_norm.weight":
			data, err := io.ReadAll(tensor.Data)
			if err != nil {
				t.Fatal(err)
			}
			if len(data) != 8*4 || math.Float32frombits(binary.LittleEndian.Uint32(data)) != 0 {
				t.Fatalf("promoted norm = %d bytes", len(data))
			}
		}
	}
}

func TestDenseTensorName(t *testing.T) {
	for source, want := range map[string]string{
		"model.embed_tokens.weight":                       "token_embd.weight",
		"model.layers.17.self_attn.q_proj.weight":         "blk.17.attn_q.weight",
		"model.layers.17.post_attention_layernorm.weight": "blk.17.ffn_norm.weight",
		"model.layers.17.mlp.down_proj.weight":            "blk.17.ffn_down.weight",
	} {
		got, include, err := denseTensorName(source)
		if err != nil || !include || got != want {
			t.Fatalf("mapping %q = %q, %t, %v; want %q", source, got, include, err, want)
		}
	}
}

func TestValidateQwen35RepositoryUsesRuntimeCatalog(t *testing.T) {
	repository := qwen35Fixture(t)
	spec, err := ValidateRepository(repository)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen35" || spec.BlockCount != 4 || spec.NextNPredictLayers != 1 ||
		spec.SSMInnerSize != 8 || spec.VocabularySize != 32 || !spec.IsRecurrentLayer(0) || spec.IsRecurrentLayer(3) {
		t.Fatalf("spec = %+v", spec)
	}
}

func TestQwen35TensorDataSqueezesConvolutionAndExcludesVision(t *testing.T) {
	tensors, err := Qwen35TensorData(qwen35Fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	foundConvolution := false
	for _, tensor := range tensors {
		if strings.HasPrefix(tensor.Name, "model.visual.") {
			t.Fatalf("vision tensor leaked into language catalog: %q", tensor.Name)
		}
		if tensor.Name == "blk.0.ssm_conv1d.weight" {
			foundConvolution = true
			if !slices.Equal(tensor.Shape, []uint64{3, 16}) {
				t.Fatalf("convolution shape = %v", tensor.Shape)
			}
		}
	}
	if !foundConvolution {
		t.Fatal("convolution mapping missing")
	}
}

func denseFixture(t *testing.T, architecture string, biases bool) *hfrepo.Repository {
	t.Helper()
	config := map[string]json.RawMessage{}
	for key, value := range map[string]any{
		"max_position_embeddings": 16,
		"hidden_size":             8,
		"num_hidden_layers":       1,
		"intermediate_size":       16,
		"num_attention_heads":     2,
		"num_key_value_heads":     1,
		"head_dim":                4,
		"rope_theta":              10000,
		"rms_norm_eps":            1e-6,
		"vocab_size":              32,
	} {
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		config[key] = encoded
	}
	shapes := map[string][]uint64{
		"model.embed_tokens.weight":                      {32, 8},
		"model.norm.weight":                              {8},
		"model.layers.0.input_layernorm.weight":          {8},
		"model.layers.0.post_attention_layernorm.weight": {8},
		"model.layers.0.self_attn.q_proj.weight":         {8, 8},
		"model.layers.0.self_attn.k_proj.weight":         {4, 8},
		"model.layers.0.self_attn.v_proj.weight":         {4, 8},
		"model.layers.0.self_attn.o_proj.weight":         {8, 8},
		"model.layers.0.mlp.gate_proj.weight":            {16, 8},
		"model.layers.0.mlp.up_proj.weight":              {16, 8},
		"model.layers.0.mlp.down_proj.weight":            {8, 16},
	}
	if biases {
		shapes["model.layers.0.self_attn.q_proj.bias"] = []uint64{8}
		shapes["model.layers.0.self_attn.k_proj.bias"] = []uint64{4}
		shapes["model.layers.0.self_attn.v_proj.bias"] = []uint64{4}
	}
	source := &safetensors.Source{Tensors: make(map[string]safetensors.Tensor)}
	for name, shape := range shapes {
		source.Tensors[name] = testTensor(t, name, shape)
	}
	return &hfrepo.Repository{
		Directory: "fixture", Config: config,
		Identity: hfrepo.Identity{ModelType: architecture}, Tensors: source,
	}
}

func qwen35Fixture(t *testing.T) *hfrepo.Repository {
	t.Helper()
	text := map[string]any{
		"max_position_embeddings": 16,
		"hidden_size":             8,
		"num_hidden_layers":       4,
		"mtp_num_hidden_layers":   1,
		"intermediate_size":       16,
		"num_attention_heads":     2,
		"num_key_value_heads":     1,
		"head_dim":                4,
		"rms_norm_eps":            1e-6,
		"vocab_size":              32,
		"full_attention_interval": 4,
		"linear_conv_kernel_dim":  3,
		"linear_key_head_dim":     2,
		"linear_num_key_heads":    2,
		"linear_num_value_heads":  4,
		"linear_value_head_dim":   2,
		"layer_types": []string{
			"linear_attention", "linear_attention", "linear_attention", "full_attention",
		},
		"rope_parameters": map[string]any{
			"rope_theta": 10000, "partial_rotary_factor": 1, "mrope_section": []int32{1, 1, 0},
		},
	}
	encodedText, err := json.Marshal(text)
	if err != nil {
		t.Fatal(err)
	}
	config := map[string]json.RawMessage{"text_config": encodedText}
	shapes := map[string][]uint64{
		"model.language_model.embed_tokens.weight": {32, 8},
		"model.language_model.norm.weight":         {8},
		"model.visual.patch_embed.weight":          {8, 8},
		"mtp.fc.weight":                            {8, 16},
		"mtp.pre_fc_norm_embedding.weight":         {8},
		"mtp.pre_fc_norm_hidden.weight":            {8},
		"mtp.norm.weight":                          {8},
	}
	for block := 0; block < 5; block++ {
		prefix := fmt.Sprintf("model.language_model.layers.%d.", block)
		if block == 4 {
			prefix = "mtp.layers.0."
		}
		shapes[prefix+"input_layernorm.weight"] = []uint64{8}
		shapes[prefix+"post_attention_layernorm.weight"] = []uint64{8}
		shapes[prefix+"mlp.gate_proj.weight"] = []uint64{16, 8}
		shapes[prefix+"mlp.up_proj.weight"] = []uint64{16, 8}
		shapes[prefix+"mlp.down_proj.weight"] = []uint64{8, 16}
		if block < 3 {
			shapes[prefix+"linear_attn.in_proj_qkv.weight"] = []uint64{16, 8}
			shapes[prefix+"linear_attn.in_proj_z.weight"] = []uint64{8, 8}
			shapes[prefix+"linear_attn.in_proj_a.weight"] = []uint64{4, 8}
			shapes[prefix+"linear_attn.in_proj_b.weight"] = []uint64{4, 8}
			shapes[prefix+"linear_attn.A_log"] = []uint64{4}
			shapes[prefix+"linear_attn.dt_bias"] = []uint64{4}
			shapes[prefix+"linear_attn.conv1d.weight"] = []uint64{16, 1, 3}
			shapes[prefix+"linear_attn.norm.weight"] = []uint64{2}
			shapes[prefix+"linear_attn.out_proj.weight"] = []uint64{8, 8}
			continue
		}
		shapes[prefix+"self_attn.q_proj.weight"] = []uint64{16, 8}
		shapes[prefix+"self_attn.k_proj.weight"] = []uint64{4, 8}
		shapes[prefix+"self_attn.v_proj.weight"] = []uint64{4, 8}
		shapes[prefix+"self_attn.o_proj.weight"] = []uint64{8, 8}
		shapes[prefix+"self_attn.q_norm.weight"] = []uint64{4}
		shapes[prefix+"self_attn.k_norm.weight"] = []uint64{4}
	}
	source := &safetensors.Source{Tensors: make(map[string]safetensors.Tensor)}
	for name, shape := range shapes {
		source.Tensors[name] = testTensor(t, name, shape)
	}
	return &hfrepo.Repository{
		Directory: "qwen35-fixture", Config: config,
		Identity: hfrepo.Identity{ModelType: "qwen3_5", TextModelType: "qwen3_5_text"},
		Tensors:  source,
	}
}

func testTensor(t *testing.T, name string, shape []uint64) safetensors.Tensor {
	t.Helper()
	size, err := safetensors.TensorBytes("BF16", shape)
	if err != nil {
		t.Fatal(err)
	}
	tensor, err := safetensors.NewTensor(name, "BF16", shape, bytes.NewReader(make([]byte, size)), 0, int64(size))
	if err != nil {
		t.Fatal(err)
	}
	return tensor
}
