package inference

import (
	"bytes"
	"fmt"
	"sync"
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

func TestGrammarVocabularyBindingReuse(t *testing.T) {
	newRunner := func(size int) *Runner {
		vocab := &tokenizer.Vocab{
			Model: "llama", Tokens: make([]tokenizer.Token, size),
			BOS: tokenizer.NullToken, EOS: tokenizer.TokenID(size - 1),
			EOT: tokenizer.NullToken, EOM: tokenizer.NullToken, UNK: tokenizer.NullToken,
			SEP: tokenizer.NullToken, PAD: tokenizer.NullToken, Mask: tokenizer.NullToken,
		}
		for i := range vocab.Tokens {
			vocab.Tokens[i] = tokenizer.Token{Text: fmt.Sprint(i), Type: tokenizer.TokenNormal}
		}
		return &Runner{preparedModel: preparedModel{vocab: vocab}}
	}
	assertSample := func(runner *Runner, source string, want int) {
		t.Helper()
		grammar, err := runner.CompileGBNF(source, "")
		if err != nil {
			t.Error(err)
			return
		}
		sampler, err := sampling.New(sampling.Config{GBNF: grammar})
		if err != nil {
			t.Error(err)
			return
		}
		logits := make([]float32, 8)
		for _, want := range []int{want, 7} {
			got, err := sampler.Sample(logits)
			if err != nil || got != want {
				t.Errorf("sample = %d, %v; want %d", got, err, want)
			}
		}
	}

	t.Run("concurrent first binding", func(t *testing.T) {
		runner := newRunner(8)
		var callers sync.WaitGroup
		for token := range 7 {
			callers.Go(func() { assertSample(runner, fmt.Sprintf("root ::= %q", fmt.Sprint(token)), token) })
		}
		callers.Wait()
		binding, err := runner.grammarVocabulary()
		if err != nil {
			t.Fatal(err)
		}
		// A second decode of ANY token would now fail. This probe makes zero
		// second-compile DecodePiece calls observable without runtime test hooks.
		for i := range runner.vocab.Tokens {
			runner.vocab.Tokens[i] = tokenizer.Token{Text: "invalid-byte", Type: tokenizer.TokenByte}
		}
		assertSample(runner, `root ::= "1"`, 1)
		reused, err := runner.grammarVocabulary()
		if err != nil || binding == nil || reused != binding {
			t.Fatal("binding was not retained")
		}
		if _, err := runner.CompileLazyGBNF(`root ::= "0"`, "", nil, []tokenizer.TokenID{0}); err != nil {
			t.Fatal(err)
		}
		other := newRunner(8)
		other.vocab.Tokens[1].Text = "different"
		assertSample(other, `root ::= "different"`, 1)
		otherBinding, err := other.grammarVocabulary()
		if err != nil {
			t.Fatal(err)
		}
		if otherBinding == binding {
			t.Fatal("different runners shared a vocabulary binding")
		}
	})

	t.Run("compile failure does not poison vocabulary", func(t *testing.T) {
		runner := newRunner(8)
		if _, err := runner.CompileGBNF(`root ::= missing`, ""); err == nil {
			t.Fatal("invalid grammar accepted")
		}
		assertSample(runner, `root ::= "0"`, 0)
		bad := newRunner(8)
		bad.vocab.Tokens[1] = tokenizer.Token{Text: "invalid-byte", Type: tokenizer.TokenByte}
		for range 2 {
			if _, err := bad.CompileGBNF(`root ::= "0"`, ""); err == nil {
				t.Fatal("failed decode published a partial binding")
			}
			if value, err := bad.grammarVocabulary(); value != nil || err == nil {
				t.Fatal("failed decode published a partial binding")
			}
		}
	})

	t.Run("warm compile allocation and state parity", func(t *testing.T) {
		var baseline float64
		for _, size := range []int{8, 8192} {
			runner := newRunner(size)
			first, err := runner.CompileGBNF(`root ::= "0"`, "")
			if err != nil {
				t.Fatal(err)
			}
			allocations := testing.AllocsPerRun(10, func() {
				if _, err := runner.CompileGBNF(`root ::= "0"`, ""); err != nil {
					t.Fatal(err)
				}
			})
			t.Logf("vocabulary=%d warm compile allocations=%.0f", size, allocations)
			if baseline != 0 && allocations != baseline {
				t.Fatalf("allocations grew: %.0f -> %.0f", baseline, allocations)
			}
			baseline = allocations
			second, err := runner.CompileGBNF(`root ::= "0"`, "")
			if err != nil {
				t.Fatal(err)
			}
			var previous []byte
			for _, grammar := range []*sampling.GBNFGrammar{first, second} {
				sampler, err := sampling.New(sampling.Config{GBNF: grammar, Seed: 17})
				if err != nil {
					t.Fatal(err)
				}
				state, err := sampler.SaveState()
				if err != nil {
					t.Fatal(err)
				}
				if previous != nil && !bytes.Equal(previous, state) {
					t.Fatal("warm signature changed")
				}
				previous = state
			}
		}
	})
}

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
