package inference

import (
	"context"
	"strings"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tokenizer"
)

func TestGenerateT5Admission(t *testing.T) {
	if _, _, _, err := (*Runner)(nil).GenerateT5(context.Background(), "", GenerateOptions{}); err == nil {
		t.Fatal("nil runner was accepted")
	}
	vocab := &tokenizer.Vocab{}
	wrong := &Runner{spec: model.Spec{Architecture: "llama"}, vocab: vocab}
	if _, _, _, err := wrong.GenerateT5(context.Background(), "", GenerateOptions{}); err == nil ||
		!strings.Contains(err.Error(), "requires T5 architecture") {
		t.Fatalf("architecture error = %v", err)
	}
	runner := &Runner{spec: model.Spec{Architecture: "t5"}, vocab: vocab}
	if _, _, _, err := runner.GenerateT5(context.Background(), "", GenerateOptions{MaxNewTokens: -1}); err == nil {
		t.Fatal("negative token limit was accepted")
	}
	if _, _, _, err := runner.GenerateT5(context.Background(), "", GenerateOptions{MinCacheReuse: -1}); err == nil ||
		!strings.Contains(err.Error(), "minimum cache reuse") {
		t.Fatalf("minimum-cache error = %v", err)
	}
	if _, _, _, err := runner.GenerateT5(context.Background(), "", GenerateOptions{ProjectedInputs: &ProjectedInputs{}}); err == nil ||
		!strings.Contains(err.Error(), "projected decoder inputs") {
		t.Fatalf("projected-input error = %v", err)
	}
	if _, _, err := runner.Generate(context.Background(), "", GenerateOptions{MaxNewTokens: -1}); err == nil ||
		!strings.Contains(err.Error(), "max new tokens") {
		t.Fatalf("generic T5 dispatch error = %v", err)
	}
}
