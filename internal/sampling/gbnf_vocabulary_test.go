package sampling

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"sync"
	"testing"
)

func TestGBNFVocabularyBindingShared(t *testing.T) {
	pieces := bytePieces("a", "b", "\xc3", "\xa9", "")
	eos := []int{4}
	names := map[string]int{"<special>": 0}
	vocabulary, err := NewGBNFVocabulary(pieces, eos, names)
	if err != nil {
		t.Fatal(err)
	}
	// Callers retain their inputs; the binding must own all mutable containers.
	pieces[0][0] = 'z'
	pieces[1] = []byte("changed")
	eos[0] = 0
	names["<special>"] = 1

	for _, test := range []struct {
		source string
		tokens []int
		state  string
	}{
		{`root ::= "a"`, []int{0, 4}, "4c3247534d503034e7cf6612ca92d1e01100000000000000000000000000000000000000000000000000000000000000000000000000000000000000"},
		{`root ::= <special> "b"`, []int{0, 1, 4}, "4c3247534d5030343c3edb0151a0aaa91100000000000000000000000000000000000000000000000000000000000000000000000000000000000000"},
		{`root ::= "é"`, []int{2, 3, 4}, "4c3247534d50303495e8e3148c12f51a1100000000000000000000000000000000000000000000000000000000000000000000000000000000000000"},
	} {
		t.Run(test.source, func(t *testing.T) {
			grammar, err := vocabulary.Compile(test.source, "", GBNFLazyOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if &grammar.tokenPieces[0] != &vocabulary.pieces[0] || &grammar.eos[0] != &vocabulary.eos[0] {
				t.Fatal("compilation copied immutable vocabulary")
			}
			// Captured with the pre-change constructor at 5f971350. This pins
			// the persisted signature independently of either new API path.
			legacyState, err := hex.DecodeString(test.state)
			if err != nil {
				t.Fatal(err)
			}
			sampler, err := New(Config{GBNF: grammar, Seed: 17})
			if err != nil {
				t.Fatal(err)
			}
			state, err := sampler.SaveState()
			if err != nil || !bytes.Equal(state, legacyState) {
				t.Fatalf("saved state = %x, %v; want pre-change %x", state, err, legacyState)
			}
			if err := sampler.LoadState(legacyState); err != nil {
				t.Fatal(err)
			}
			for _, want := range test.tokens {
				got, err := sampler.Sample([]float32{1, 2, 3, 4, 100})
				if err != nil || got != want {
					t.Fatalf("sample = %d, %v; want %d", got, err, want)
				}
			}
			if !gbnfAcceptsTokens(t, grammar, test.tokens) || gbnfAcceptsTokens(t, grammar, []int{4}) {
				t.Fatal("fresh grammar state or EOS acceptance changed")
			}
		})
	}

	t.Run("concurrent distinct and lazy grammars", func(t *testing.T) {
		var workers sync.WaitGroup
		for _, source := range []string{`root ::= "a"`, `root ::= "b"`, `root ::= "é"`} {
			workers.Go(func() {
				for range 8 {
					grammar, err := vocabulary.Compile(source, "", GBNFLazyOptions{})
					if err != nil {
						t.Error(err)
						return
					}
					if gbnfAcceptsTokens(t, grammar, []int{0, 4}) != (source == `root ::= "a"`) {
						t.Error("distinct grammar state contaminated")
					}
				}
			})
		}
		lazy, err := vocabulary.Compile(`root ::= "ab"`, "", GBNFLazyOptions{Enabled: true, Tokens: []int{0}})
		if err != nil {
			t.Fatal(err)
		}
		for range 8 {
			workers.Go(func() {
				if !gbnfAcceptsTokens(t, lazy, []int{1, 0, 1, 4}) || gbnfAcceptsTokens(t, lazy, []int{0, 0, 4}) {
					t.Error("shared lazy grammar leaked trigger/parse state")
				}
			})
		}
		workers.Wait()
	})

	t.Run("allocations independent of vocabulary size", func(t *testing.T) {
		var baseline float64
		for _, size := range []int{8, 8192} {
			pieces := make([][]byte, size)
			for i := range pieces {
				pieces[i] = []byte(fmt.Sprint(i))
			}
			vocabulary, err := NewGBNFVocabulary(pieces, []int{size - 1}, nil)
			if err != nil {
				t.Fatal(err)
			}
			allocations := testing.AllocsPerRun(10, func() {
				if _, err := vocabulary.Compile(`root ::= "0"`, "", GBNFLazyOptions{}); err != nil {
					t.Fatal(err)
				}
			})
			t.Logf("vocabulary=%d compile allocations=%.0f", size, allocations)
			if baseline != 0 && allocations != baseline {
				t.Fatalf("compile allocations grew from %.0f to %.0f", baseline, allocations)
			}
			baseline = allocations
		}
	})

	t.Run("invalid input retains binding", func(t *testing.T) {
		if _, err := NewGBNFVocabulary(nil, nil, nil); err == nil {
			t.Fatal("empty vocabulary accepted")
		}
		for _, invalid := range []*GBNFVocabulary{nil, {}} {
			if _, err := invalid.Compile(`root ::= "a"`, "", GBNFLazyOptions{}); err == nil {
				t.Fatal("uninitialized vocabulary accepted")
			}
		}
		if _, err := vocabulary.Compile(`root ::= missing`, "", GBNFLazyOptions{}); err == nil {
			t.Fatal("invalid grammar accepted")
		}
		valid, err := vocabulary.Compile(`root ::= "b"`, "", GBNFLazyOptions{})
		if err != nil || !gbnfAcceptsTokens(t, valid, []int{1, 4}) {
			t.Fatalf("invalid compile damaged binding: %v", err)
		}
	})
}
