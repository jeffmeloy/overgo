package tokenizer

import (
	"os"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testevidence"
)

func TestUGMViterbiPrefersHighestSequenceScore(t *testing.T) {
	vocab := &Vocab{
		Model:     "t5",
		Tokens:    []Token{{Text: "<pad>", Type: TokenControl}, {Text: "</s>", Type: TokenControl}, {Text: "<unk>", Type: TokenUnknown}, {Text: "▁", Score: -1, Type: TokenNormal}, {Text: "h", Score: -1, Type: TokenNormal}, {Text: "i", Score: -1, Type: TokenNormal}, {Text: "▁hi", Score: -0.5, Type: TokenNormal}},
		EOS:       1,
		UNK:       2,
		AddEOS:    true,
		AddPrefix: true,
		tokenToID: map[string]TokenID{"▁": 3, "h": 4, "i": 5, "▁hi": 6},
		ugmMaxLen: len("▁hi"),
	}
	got, err := vocab.Encode("hi", EncodeOptions{AddSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	if want := []TokenID{6, 1}; !slices.Equal(got, want) {
		t.Fatalf("UGM IDs = %v, want %v", got, want)
	}
}

func TestRealUMT5Tokenization(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": requires a real UMT5 model")
	}
	path := os.Getenv("OVERGO_UMT5_MODEL")
	if path == "" {
		t.Skip("OVERGO_UMT5_MODEL is not set")
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	vocab, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	got, err := vocab.Encode("Hello world!", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if want := []TokenID{23231, 3914, 332}; !slices.Equal(got, want) {
		t.Fatalf("UMT5 IDs = %v, want pinned llama.cpp %v", got, want)
	}
}
