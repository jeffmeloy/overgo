package testutil

import (
	_ "embed"
	"encoding/json"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/safetensors"
)

// qwen35DirectoryJSON is a Qwen 3.5 Hugging Face directory at the smallest shapes the converter's tables take: its
// configuration, a tokenizer of 32 tokens, and the shape of each weight, among them every part the converter transforms
// (the norms it shifts by one, A_log, the value heads it reorders, the convolution it squeezes) and tensors it keeps as
// they are.
//
//go:embed qwen35_directory.json
var qwen35DirectoryJSON []byte

// qwen35WeightScale keeps the fixture's weights within a quarter of zero, the size of an initialized model's.
const qwen35WeightScale = 0.25

// Qwen35Directory writes the directory qwen35DirectoryJSON describes, each weight value distinct, and returns it.
func Qwen35Directory(t testing.TB) string {
	t.Helper()
	var document struct {
		Config    json.RawMessage  `json:"config"`
		Tokenizer json.RawMessage  `json:"tokenizer"`
		Shapes    map[string][]int `json:"shapes"`
	}
	if err := json.Unmarshal(qwen35DirectoryJSON, &document); err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	WriteTextFile(t, directory, "config.json", string(document.Config))
	WriteTextFile(t, directory, "tokenizer.json", string(document.Tokenizer))
	weights := map[string][]float32{}
	seed := 0
	for name, shape := range document.Shapes {
		elements := 1
		for _, dimension := range shape {
			elements *= dimension
		}
		values := make([]float32, elements)
		for index := range values {
			seed++
			values[index] = float32(math.Sin(float64(seed)) * qwen35WeightScale)
		}
		weights[name] = values
	}
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, document.Shapes, nil); err != nil {
		t.Fatal(err)
	}
	return directory
}
