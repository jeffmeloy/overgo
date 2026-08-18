package inference

import (
	"slices"
	"testing"

	"overgo/internal/tokenizer"
)

func TestSamplingInfillVocabularyUsesUnrenderedSpecialPieces(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{vocab: &tokenizer.Vocab{
		Model: "llama",
		Tokens: []tokenizer.Token{
			{Text: "a", Type: tokenizer.TokenNormal},
			{Text: "<fim-prefix>", Type: tokenizer.TokenControl},
			{Text: "</s>", Type: tokenizer.TokenControl},
		},
		BOS:  tokenizer.NullToken,
		EOS:  2,
		EOT:  tokenizer.NullToken,
		EOM:  tokenizer.NullToken,
		UNK:  tokenizer.NullToken,
		SEP:  tokenizer.NullToken,
		PAD:  tokenizer.NullToken,
		Mask: tokenizer.NullToken,
	}}}
	vocabulary, err := runner.SamplingInfillVocabulary()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(vocabulary.Pieces, []string{"a", "", ""}) {
		t.Fatalf("infill pieces = %q", vocabulary.Pieces)
	}
	if !slices.Equal(vocabulary.EOG, []bool{false, false, true}) {
		t.Fatalf("infill EOG table = %v", vocabulary.EOG)
	}
	if vocabulary.EOT != -1 || vocabulary.EOS != 2 {
		t.Fatalf("infill EOT/EOS = %d/%d", vocabulary.EOT, vocabulary.EOS)
	}
	ids := []tokenizer.TokenID{0, 1}
	text, err := runner.Detokenize(ids, RenderText)
	if err != nil || text != "a" {
		t.Fatalf("text render = %q, %v", text, err)
	}
	prompt, err := runner.Detokenize(ids, RenderPrompt)
	if err != nil || prompt != "a<fim-prefix>" {
		t.Fatalf("prompt render = %q, %v", prompt, err)
	}
	if _, err := runner.Detokenize(ids, RenderPrompt+1); err == nil {
		t.Fatal("invalid render mode accepted")
	}
}
