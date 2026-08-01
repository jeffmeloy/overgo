package inference

import (
	"context"
	"math"
	"strings"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

func TestCogVLMRejectsVisualEmbeddingMode(t *testing.T) {
	runner := &Runner{spec: model.Spec{Architecture: "cogvlm"}}
	_, _, err := runner.forwardCachedWithEmbeddingOverridesLocked(
		context.Background(), nil, nil,
		[]EmbeddingOverride{{TokenIndex: 0, Embedding: []float32{1}}},
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "visual embedding mode") {
		t.Fatalf("error = %v", err)
	}
}

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

func TestMultimodalInputAdmission(t *testing.T) {
	for _, architecture := range []string{"qwen3vl", "glm4", "hunyuan-dense"} {
		supported := &Runner{spec: model.Spec{
			Architecture: architecture, RopeSections: [4]int32{2, 2, 0, 0},
		}}
		_, _, err := supported.ForwardCachedWithMultimodalInputs(
			context.Background(), nil, nil, MultiAxisPositions{}, nil,
		)
		if err == nil || !strings.Contains(err.Error(), "token sequence is empty") {
			t.Fatalf("%s multimodal error = %v", architecture, err)
		}
	}

	unsupported := &Runner{spec: model.Spec{Architecture: "llama"}}
	_, _, err := unsupported.ForwardCachedWithMultimodalInputs(
		context.Background(), nil, nil, MultiAxisPositions{}, nil,
	)
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("unsupported multimodal error = %v", err)
	}
}

func TestMultimodalInputRejectsIncompleteAxes(t *testing.T) {
	runner := &Runner{spec: model.Spec{
		Architecture: "qwen3vl", ContextLength: 4,
		RopeSections: [4]int32{1, 1, 0, 0},
	}}
	positions := MultiAxisPositions{{0}, {0}, nil, {0}}
	_, _, err := runner.ForwardCachedWithMultimodalInputs(
		context.Background(), []tokenizer.TokenID{0}, nil, positions, nil,
	)
	if err == nil || !strings.Contains(err.Error(), "position 2") {
		t.Fatalf("incomplete multi-axis error = %v", err)
	}
}
