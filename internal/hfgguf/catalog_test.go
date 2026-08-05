package hfgguf

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
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
