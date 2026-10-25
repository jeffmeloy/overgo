package evaluation

import (
	"math/rand/v2"
	"strings"
)

// Strata's fresh long prompt (ab_engine.long_prompt in its tools at
// 296b59d2): the seed of the generator its words are drawn by, and the words
// they are drawn from.
const wordListSeed = 7

var wordListVocabulary = strings.Fields("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz meadow anchor violet thunder")

// WordList is Strata's fresh long prompt of count words: a list of words
// drawn by a generator seeded as Strata seeds its own, between its opening
// and its question, so a prompt of one size reads alike on every engine and
// its size, not its words, sets the rate it is read at.
func WordList(count int) string {
	draws := rand.New(rand.NewPCG(wordListSeed, wordListSeed))
	words := make([]string, count)
	for index := range words {
		words[index] = wordListVocabulary[draws.IntN(len(wordListVocabulary))]
	}
	return "Here is a list of words:\n" + strings.Join(words, " ") + "\n\nSummarize what kinds of words appear in the list, in five sentences."
}

// Strata's two sizes of fresh prompt, in tokens.
const (
	shortPromptTokens = 4096
	longPromptTokens  = 16384
)

// PromptLengths are the longer prompts, in tokens, a chat model's first
// output is read at beside its verified request: Strata's two sizes of fresh
// prompt, so a model's reading rate past a short request is measured where
// a reference engine's is.
var PromptLengths = []int{shortPromptTokens, longPromptTokens}
