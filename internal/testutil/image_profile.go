package testutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
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
