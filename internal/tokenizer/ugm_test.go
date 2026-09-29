package tokenizer

import (
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/modeltest"
	"overgo/internal/testskip"
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

// TestRealUMT5Tokenization holds the validated UMT5-XXL encoder fixture's
// vocabulary to llama.cpp's token IDs.
func TestRealUMT5Tokenization(t *testing.T) {
	testskip.Short(t, "requires a real UMT5 model")
	file, err := gguf.Open(modeltest.Directory(t, "umt5-xxl-encoder-Q5_K_S.gguf"))
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
