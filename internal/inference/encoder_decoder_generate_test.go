package inference

import (
	"strings"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tokenizer"
)

func TestEncoderDecoderGenerationAdmission(t *testing.T) {
	if _, _, _, err := (*Runner)(nil).GenerateEncoderDecoder(t.Context(), "", GenerateOptions{}); err == nil {
		t.Fatal("nil runner was accepted")
	}
	vocab := &tokenizer.Vocab{}
	wrong := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}, vocab: vocab}}
	wrong = attachFixtureProgram(wrong)
	if _, _, _, err := wrong.GenerateEncoderDecoder(t.Context(), "", GenerateOptions{}); err == nil ||
		!strings.Contains(err.Error(), "compiled encoder-decoder program") {
		t.Fatalf("architecture error = %v", err)
	}
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "t5"}}, vocab: vocab}}
	runner = attachFixtureProgram(runner)
	if _, _, _, err := runner.GenerateEncoderDecoder(t.Context(), "", GenerateOptions{MaxNewTokens: -1}); err == nil {
		t.Fatal("negative token limit was accepted")
	}
	if _, _, _, err := runner.GenerateEncoderDecoder(t.Context(), "", GenerateOptions{MinCacheReuse: -1}); err == nil ||
		!strings.Contains(err.Error(), "minimum cache reuse") {
		t.Fatalf("minimum-cache error = %v", err)
	}
	if _, _, _, err := runner.GenerateEncoderDecoder(t.Context(), "", GenerateOptions{ProjectedInputs: &ProjectedInputs{}}); err == nil ||
		!strings.Contains(err.Error(), "projected decoder inputs") {
		t.Fatalf("projected-input error = %v", err)
	}
	if _, _, err := runner.Generate(t.Context(), "", GenerateOptions{MaxNewTokens: -1}); err == nil ||
		!strings.Contains(err.Error(), "max new tokens") {
		t.Fatalf("generic encoder-decoder dispatch error = %v", err)
	}
}
