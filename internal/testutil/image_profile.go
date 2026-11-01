package testutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/safetensors"
)

// WriteImageProfileFixture writes one complete profile-source artifact.
func WriteImageProfileFixture(t testing.TB, root string) {
	t.Helper()
	writeJSONFixture(t, filepath.Join(root, "model_index.json"), map[string]any{
		"_class_name":  "Krea2Pipeline",
		"scheduler":    []string{"diffusers", "FlowMatchEulerDiscreteScheduler"},
		"text_encoder": []string{"transformers", "Qwen3VLModel"},
		"tokenizer":    []string{"transformers", "Qwen2Tokenizer"},
		"transformer":  []string{"diffusers", "Krea2Transformer2DModel"},
		"vae":          []string{"diffusers", "AutoencoderKLQwenImage"},
	})
	writeJSONFixture(t, filepath.Join(root, "scheduler", "scheduler_config.json"), map[string]any{
		"_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256,
		"base_shift": 0.5, "max_image_seq_len": 6400, "max_shift": 1.15,
		"num_train_timesteps": 1000, "time_shift_type": "exponential", "use_dynamic_shifting": true,
	})
	writeJSONFixture(t, filepath.Join(root, "tokenizer", "tokenizer_config.json"), map[string]any{
		"tokenizer_class": "Qwen2Tokenizer", "pad_token": "<|endoftext|>",
	})
}

// imageFixturePatchSize follows Krea2Pipeline's latent patch geometry.
// Reconsider it when the structural fixture adopts another source profile.
const imageFixturePatchSize = 2

// WriteImageInventoryFixture writes a sharded Krea source for structural intake tests.
// Scalar weights establish inventory behavior and provide no numerical evidence.
func WriteImageInventoryFixture(t testing.TB, root string) {
	t.Helper()
	WriteImageProfileFixture(t, root)
	path := filepath.Join(root, "model_index.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var index map[string]any
	if err := json.Unmarshal(data, &index); err != nil {
		t.Fatal(err)
	}
	index["text_encoder_select_layers"] = []int{1}
	index["is_distilled"], index["patch_size"] = true, imageFixturePatchSize
	writeJSONFixture(t, path, index)
	writeJSONFixture(t, filepath.Join(root, "text_encoder", "config.json"), map[string]any{"model_type": "qwen3_vl", "architectures": []string{"Qwen3VLModel"}})
	writeJSONFixture(t, filepath.Join(root, "transformer", "config.json"), map[string]any{"_class_name": "Krea2Transformer2DModel"})
	writeJSONFixture(t, filepath.Join(root, "vae", "config.json"), map[string]any{"_class_name": "AutoencoderKLQwenImage"})
	writeJSONFixture(t, filepath.Join(root, "tokenizer", "tokenizer.json"), map[string]any{})
	files := map[string]string{
		"text_encoder/model.safetensors":                                 "embed",
		"vae/diffusion_pytorch_model.safetensors":                        "decode",
		"transformer/diffusion_pytorch_model-00001-of-00002.safetensors": "first",
		"transformer/diffusion_pytorch_model-00002-of-00002.safetensors": "second",
	}
	for file, name := range files {
		if err := safetensors.Save(filepath.Join(root, filepath.FromSlash(file)), map[string][]float32{name: {1}}, map[string][]int{name: {1}}, nil); err != nil {
			t.Fatal(err)
		}
	}
	writeJSONFixture(t, filepath.Join(root, "transformer", "diffusion_pytorch_model.safetensors.index.json"), map[string]any{"weight_map": map[string]string{"first": "diffusion_pytorch_model-00001-of-00002.safetensors", "second": "diffusion_pytorch_model-00002-of-00002.safetensors"}})
}

func writeJSONFixture(t testing.TB, path string, value any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
