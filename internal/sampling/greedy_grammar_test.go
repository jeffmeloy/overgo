package sampling

import (
	"math/rand/v2"
	"strconv"
	"testing"
	"time"

	"overgo/internal/testcheck"
)

// A greedy draw under a grammar is timed against one without, over a vocabulary of a model's size.
const (
	// grammarVocabulary is the vocabulary's size, Qwen3's order of magnitude.
	grammarVocabulary = 100_000
	// grammarDraws seeds the logits' source.
	grammarDraws = 8
	// grammarRound is how long a round draws for: a round of a few draws with no grammar passed below the clock's
	// resolution on Windows and measured 0s, which no ratio holds (2026-11-11).
	grammarRound = 20 * time.Millisecond
	// grammarRounds is how many rounds are timed, the fastest of each kept.
	grammarRounds = 5
	// grammarOverhead is the most a grammar that allows the model's choice may multiply a greedy draw's time.
	grammarOverhead = 10
)

// TestAGreedyDrawUnderAGrammarCostsAboutWhatOneWithoutDoes: a greedy draw under a grammar that allows the model's
// likeliest token, as a tool call's grammar mostly does, costs about what a draw with no grammar does, not a question
// to the grammar about every token of the vocabulary. Asking it each of them held Qwen3.8-27B's tool calls on the CPU,
// twenty minutes a reply with the GPU idle (2026-11-09).
func TestAGreedyDrawUnderAGrammarCostsAboutWhatOneWithoutDoes(t *testing.T) {
	pieces := make([][]byte, grammarVocabulary)
	for index := range pieces {
		pieces[index] = []byte("w" + strconv.Itoa(index))
	}
	end := len(pieces) - 1
	pieces[end] = nil
	vocabulary := testcheck.Must(NewGBNFVocabulary(pieces, []int{end}, nil))(t)
	grammar := testcheck.Must(vocabulary.Compile(`root ::= [a-z0-9]+`, "root", GBNFLazyOptions{}))(t)
	constrained := testcheck.Must(New(Config{GBNF: grammar}))(t)
	free := testcheck.Must(New(Config{}))(t)
	source := rand.New(rand.NewPCG(grammarVocabulary, grammarDraws))
	logits := make([]float32, len(pieces))
	for index := range logits {
		logits[index] = source.Float32()
	}
	// The end never outweighs a word, so the grammar never closes.
	logits[end] = -1
	// fastest is the fastest round's time a draw, each round drawing until grammarRound has passed.
	fastest := func(sampler *Sampler) time.Duration {
		best := time.Duration(0)
		for range grammarRounds {
			started, draws := time.Now(), 0
			for time.Since(started) < grammarRound {
				testcheck.Must(sampler.Sample(logits))(t)
				draws++
			}
			if spent := time.Since(started) / time.Duration(draws); best == 0 || spent < best {
				best = spent
			}
		}
		return best
	}
	without, under := fastest(free), fastest(constrained)
	t.Logf("a greedy draw over %d tokens: %s with no grammar, %s under one", grammarVocabulary, without, under)
	if under > grammarOverhead*without {
		t.Fatalf("a greedy draw under a grammar that allows the model's choice took %s, more than %d times one without (%s)", under, grammarOverhead, without)
	}
}
