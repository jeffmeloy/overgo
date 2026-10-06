package thoughtbank

import (
	"context"
	"math"
	"strings"
	"testing"
)

// scoringText is n characters of the fixture tokenizer's alphabet.
func scoringText(vocabulary, n int) string {
	var text strings.Builder
	for index := range n {
		text.WriteRune(rune('a' + (index*7+index/3)%vocabulary))
	}
	return text.String()
}

// TestScoreSequenceIsTheForwardsLogProbability holds a sequence's score to the
// log of the probability the forward gives each token after the first, and its
// count to the tokens scored.
func TestScoreSequenceIsTheForwardsLogProbability(t *testing.T) {
	t.Parallel()
	generator, vocabulary := fixtureGenerator(t)
	text := scoringText(vocabulary, 9)
	ids := make([]int32, len(text))
	for index, character := range text {
		ids[index] = int32(character - 'a')
	}
	output, err := FastWeightBankLMForward(ids, nil, 0, generator.weights)
	if err != nil {
		t.Fatal(err)
	}
	var want float64
	for position := 1; position < len(ids); position++ {
		logits := output.Logits[(position-1)*vocabulary : position*vocabulary]
		var total float64
		for _, logit := range logits {
			total += math.Exp(float64(logit))
		}
		want += float64(logits[ids[position]]) - math.Log(total)
	}
	scores, err := generator.ScoreContinuations(t.Context(), "", []string{text})
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 1 || scores[0].Tokens != uint64(len(ids)-1) || math.Abs(scores[0].LogProbability-want) > 1e-6 {
		t.Fatalf("score = %+v, want %d tokens and log-probability %g", scores, len(ids)-1, want)
	}
}

// TestScoreSequenceReadsALongSequenceInDisjointWindows holds a sequence longer
// than the model's window to the sum of its windows scored apart: the first
// token of each window is read and not scored.
func TestScoreSequenceReadsALongSequenceInDisjointWindows(t *testing.T) {
	t.Parallel()
	generator, vocabulary := fixtureGenerator(t)
	const tail = 5
	text := scoringText(vocabulary, 2*scoringWindowTokens+tail)
	windows := []string{text[:scoringWindowTokens], text[scoringWindowTokens : 2*scoringWindowTokens], text[2*scoringWindowTokens:]}
	whole, err := generator.ScoreContinuations(t.Context(), "", []string{text})
	if err != nil {
		t.Fatal(err)
	}
	apart, err := generator.ScoreContinuations(t.Context(), "", windows)
	if err != nil {
		t.Fatal(err)
	}
	var tokens uint64
	var logProbability float64
	for _, score := range apart {
		tokens += score.Tokens
		logProbability += score.LogProbability
	}
	if tokens != uint64(len(text)-len(windows)) || whole[0].Tokens != tokens || math.Abs(whole[0].LogProbability-logProbability) > 1e-6 {
		t.Fatalf("whole = %+v, windows sum to %d tokens and %g; want %d tokens", whole[0], tokens, logProbability, len(text)-len(windows))
	}
}

// TestScoreSequenceRefusesWhatItCannotScore holds the scorer to its contract:
// no prompt, at least one scored token, and a live context.
func TestScoreSequenceRefusesWhatItCannotScore(t *testing.T) {
	t.Parallel()
	generator, vocabulary := fixtureGenerator(t)
	text := scoringText(vocabulary, 4)
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	for name, call := range map[string]func() error{
		"a prompt":            func() error { _, err := generator.ScoreContinuations(t.Context(), "ab", []string{text}); return err },
		"a one-token text":    func() error { _, err := generator.ScoreContinuations(t.Context(), "", []string{"a"}); return err },
		"a cancelled context": func() error { _, err := generator.ScoreContinuations(cancelled, "", []string{text}); return err },
	} {
		if call() == nil {
			t.Errorf("%s was scored", name)
		}
	}
}
