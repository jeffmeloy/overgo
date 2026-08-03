package inference

import (
	"testing"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

func TestCompileGBNFUsesDecodedVocabularyAndEOG(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{vocab: &tokenizer.Vocab{
		Model: "llama",
		Tokens: []tokenizer.Token{
			{Text: "a", Type: tokenizer.TokenNormal},
			{Text: "</s>", Type: tokenizer.TokenControl},
			{Text: "b", Type: tokenizer.TokenNormal},
		},
		BOS:  tokenizer.NullToken,
		EOS:  tokenizer.TokenID(1),
		EOT:  tokenizer.NullToken,
		EOM:  tokenizer.NullToken,
		UNK:  tokenizer.NullToken,
		SEP:  tokenizer.NullToken,
		PAD:  tokenizer.NullToken,
		Mask: tokenizer.NullToken,
	}}}
	grammar, err := runner.CompileGBNF(`root ::= "a"`, "")
	if err != nil {
		t.Fatal(err)
	}
	sampler, err := sampling.New(sampling.Config{GBNF: grammar})
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{1, 100, 10}
	token, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if token != 0 {
		t.Fatalf("GBNF vocabulary first token = %d, want 0", token)
	}
	token, err = sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if token != 1 {
		t.Fatalf("GBNF vocabulary terminal token = %d, want 1", token)
	}
}
