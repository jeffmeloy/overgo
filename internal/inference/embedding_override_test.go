package inference

import (
	"math"
	"strings"
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

func TestApplyEmbeddingOverrides(t *testing.T) {
	activation := reference.Value{
		Shape: tensor.MustShape(3, 3),
		Data:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	err := applyEmbeddingOverrides(&activation, []EmbeddingOverride{
		{TokenIndex: 1, Embedding: []float32{10, 11, 12}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 3, 10, 11, 12, 7, 8, 9}
	for index := range want {
		if activation.Data[index] != want[index] {
			t.Fatalf("embedding[%d] = %v, want %v", index, activation.Data[index], want[index])
		}
	}
}

func TestApplyEmbeddingOverridesRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name      string
		overrides []EmbeddingOverride
		contains  string
	}{
		{"index", []EmbeddingOverride{{TokenIndex: 2, Embedding: []float32{1, 2}}}, "out of range"},
		{"width", []EmbeddingOverride{{TokenIndex: 0, Embedding: []float32{1}}}, "width"},
		{"duplicate", []EmbeddingOverride{
			{TokenIndex: 0, Embedding: []float32{1, 2}},
			{TokenIndex: 0, Embedding: []float32{3, 4}},
		}, "duplicate"},
		{"non-finite", []EmbeddingOverride{{TokenIndex: 0, Embedding: []float32{1, float32(math.NaN())}}}, "non-finite"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			activation := reference.Value{Shape: tensor.MustShape(2, 2), Data: make([]float32, 4)}
			err := applyEmbeddingOverrides(&activation, test.overrides)
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("error = %v, want containing %q", err, test.contains)
			}
		})
	}
}
