package hfbpe

import "testing"

func TestDecodeStrictRejectsUnknownToken(t *testing.T) {
	tokenizer := &Tokenizer{vocab: map[string]int{"x": 3}, special: map[string]int{}}
	tokenizer.buildByteAlphabet()
	tokenizer.buildID2Vocab()
	if decoded, err := tokenizer.DecodeStrict([]int{4}); err == nil || decoded != "" {
		t.Fatalf("strict decode = %q, %v", decoded, err)
	}
	if decoded, err := tokenizer.DecodeStrict([]int{3}); err != nil || decoded != "x" {
		t.Fatalf("strict decode = %q, %v", decoded, err)
	}
}
