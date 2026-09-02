package tokenizer

import "testing"

// The upstream rule: a token named like an end marker ends generation
// whatever the metadata declares, and an undeclared end-of-turn token
// is discovered by name in upstream precedence.
func TestEndOfGenerationByNameFollowsUpstream(t *testing.T) {
	vocab := &Vocab{
		Tokens: []Token{{Text: "ordinary"}, {Text: "<|endoftext|>"}, {Text: "<|im_end|>"}, {Text: "<turn|>"}, {Text: "B"}},
		EOS:    1,
		EOT:    NullToken,
		EOM:    NullToken,
		FIMPad: NullToken,
		FIMRep: NullToken,
		FIMSep: NullToken,
		tokenToID: map[string]TokenID{
			"ordinary": 0, "<|endoftext|>": 1, "<|im_end|>": 2, "<turn|>": 3, "B": 4,
		},
	}
	vocab.markEndOfGenerationByName()
	if vocab.EOT != 2 {
		t.Fatalf("EOT = %d, want <|im_end|> (2) discovered by name", vocab.EOT)
	}
	for id, want := range map[TokenID]bool{0: false, 1: true, 2: true, 3: true, 4: false} {
		if vocab.IsEOG(id) != want {
			t.Fatalf("IsEOG(%d) = %v, want %v", id, !want, want)
		}
	}
	declared := &Vocab{
		Tokens:    []Token{{Text: "<|im_end|>"}, {Text: "<|eot_id|>"}},
		EOS:       NullToken,
		EOT:       0,
		EOM:       NullToken,
		FIMPad:    NullToken,
		FIMRep:    NullToken,
		FIMSep:    NullToken,
		tokenToID: map[string]TokenID{"<|im_end|>": 0, "<|eot_id|>": 1},
	}
	declared.markEndOfGenerationByName()
	if declared.EOT != 0 || !declared.IsEOG(1) {
		t.Fatalf("declared EOT %d overridden or named marker %v not EOG", declared.EOT, declared.IsEOG(1))
	}
}
