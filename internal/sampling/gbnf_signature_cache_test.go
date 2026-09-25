package sampling

import (
	"bytes"
	"fmt"
	"sync"
	"testing"
)

func TestGBNFVocabularyHashReuse(t *testing.T) {
	pieces := make([][]byte, 8192)
	pieces[0], pieces[1] = []byte("a"), []byte("b")
	for index := 2; index < len(pieces); index++ {
		pieces[index] = []byte(fmt.Sprint(index))
	}
	newBinding := func() *GBNFVocabulary {
		t.Helper()
		vocabulary, err := NewGBNFVocabulary(pieces, []int{len(pieces) - 1}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return vocabulary
	}
	state := func(grammar *GBNFGrammar) []byte {
		t.Helper()
		sampler, err := New(Config{GBNF: grammar, Seed: 17})
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := sampler.SaveState()
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	vocabulary := newBinding()
	const source = `root ::= "a"`
	first, err := vocabulary.Compile(source, "", GBNFLazyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	wantState := state(first)
	for range 8 {
		warm, err := vocabulary.Compile(source, "", GBNFLazyOptions{})
		if err != nil || !bytes.Equal(state(warm), wantState) {
			t.Fatalf("warm grammar changed persisted state: %v", err)
		}
	}
	if len(vocabulary.signatures) != 1 || vocabulary.signatureUsed != 1 {
		t.Fatal("identical grammar signatures were not retained once")
	}
	fresh, err := newBinding().Compile(source, "", GBNFLazyOptions{})
	if err != nil || !bytes.Equal(state(fresh), wantState) {
		t.Fatalf("reused signature differs from fresh binding: %v", err)
	}
	lazy := GBNFLazyOptions{Enabled: true, Tokens: []int{1}}
	lazyGrammar, err := vocabulary.Compile(source, "", lazy)
	if err != nil {
		t.Fatal(err)
	}
	freshLazy, err := newBinding().Compile(source, "", lazy)
	if err != nil || !bytes.Equal(state(lazyGrammar), state(freshLazy)) || bytes.Equal(state(lazyGrammar), wantState) {
		t.Fatalf("lazy options collided with ordinary grammar: %v", err)
	}
	if _, err := vocabulary.Compile(`root ::= missing`, "", GBNFLazyOptions{}); err == nil || len(vocabulary.signatures) != 2 {
		t.Fatal("failed compile entered the signature cache")
	}
	for index := range gbnfSignatureCacheCapacity + 9 {
		source := fmt.Sprintf("root ::= \"a\"%s", bytes.Repeat([]byte{' '}, index))
		grammar, err := vocabulary.Compile(source, "", GBNFLazyOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if freshGrammar, err := newBinding().Compile(source, "", GBNFLazyOptions{}); err != nil || !bytes.Equal(state(grammar), state(freshGrammar)) {
			t.Fatalf("grammar %d differs from fresh binding: %v", index, err)
		}
	}
	if len(vocabulary.signatures) != gbnfSignatureCacheCapacity || vocabulary.signatureUsed != gbnfSignatureCacheCapacity {
		t.Fatalf("signature cache grew beyond its bound: %d", len(vocabulary.signatures))
	}
	revisited, err := vocabulary.Compile(source, "", GBNFLazyOptions{})
	if err != nil || !bytes.Equal(state(revisited), wantState) {
		t.Fatalf("evicted grammar changed legacy state: %v", err)
	}
	var workers sync.WaitGroup
	for index := range 12 {
		workers.Go(func() {
			source := fmt.Sprintf("root ::= \"%s\"", []string{"a", "b"}[index%2])
			for range 8 {
				grammar, err := vocabulary.Compile(source, "", GBNFLazyOptions{})
				if err != nil {
					t.Error(err)
					return
				}
				if grammar.signature == 0 {
					t.Error("concurrent compile lost signature")
				}
			}
		})
	}
	workers.Wait()
}
