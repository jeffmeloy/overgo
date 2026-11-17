package inference

import (
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

// TestPenaltiesReadFromWhereTheGenerationSays holds GenerateOptions.PenaltiesFrom (moss-ultra-sections/prompt-modes):
// a presence penalty over the whole history pushes a prompt's token below its rival, as llama.cpp's penalties read the
// prompt, and from the prompt's end it leaves the prompt's token as it was, as ComfyUI's TextGenerate penalizes only
// what it writes; a negative start is refused.
func TestPenaltiesReadFromWhereTheGenerationSays(t *testing.T) {
	logits := []float32{1.0, 1.2}
	prompt := []tokenizer.TokenID{1}
	pick := func(from int) tokenizer.TokenID {
		t.Helper()
		sampler := testcheck.Must(sampling.New(sampling.Config{Temperature: 0, RepeatLastN: -1, RepeatPenalty: 1, PresencePenalty: 1}))(t)
		return testcheck.Must(sampleGenerationToken(append([]float32(nil), logits...), prompt, GenerateOptions{Sampler: sampler, PenaltiesFrom: from}))(t).ID
	}
	if got := pick(0); got != 0 {
		t.Errorf("penalties over the prompt chose %d, want 0 past the penalized prompt token", got)
	}
	if got := pick(len(prompt)); got != 1 {
		t.Errorf("penalties from the prompt's end chose %d, want the prompt's token 1 as it was", got)
	}
	if normalizeGenerateOptions(&GenerateOptions{PenaltiesFrom: -1}, 0) == nil {
		t.Error("a negative start of the penalties was taken")
	}
}
