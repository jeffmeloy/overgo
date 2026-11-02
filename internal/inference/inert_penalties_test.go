package inference

import (
	"testing"

	"overgo/internal/sampling"
)

// TestInertPenaltiesStayOnTheCard: a request whose penalties its window leaves inert - Qwen's presence penalty over
// the served window of 0 - decodes as a request with none does: its greedy sampler is raw greedy, which the card's
// argmax and the drafted windows serve, and its sampled one is the round's sampler's; a window of 64, over which the
// penalty applies, keeps both on the host.
func TestInertPenaltiesStayOnTheCard(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		window int
		card   bool
	}{{"window 0", 0, true}, {"window 64", 64, false}} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			greedy, err := sampling.New(sampling.Config{Temperature: 0, PresencePenalty: 1.5, RepeatPenalty: 1, RepeatLastN: test.window})
			if err != nil {
				t.Fatal(err)
			}
			if greedy.IsRawGreedy() != test.card {
				t.Fatalf("a greedy sampler with a presence penalty over a window of %d is raw greedy %t", test.window, !test.card)
			}
			drawn, err := sampling.New(sampling.Config{
				Temperature: 0.7, TopK: 20, TopP: 0.8, PresencePenalty: 1.5, RepeatPenalty: 1, RepeatLastN: test.window,
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := strataSamplingOf(GenerateOptions{Sampler: drawn}); ok != test.card {
				t.Fatalf("the round's sampler takes a draw with a presence penalty over a window of %d: %t", test.window, ok)
			}
		})
	}
}

// TestStrataRoundSamplesOnlyItsOwnChain holds the round's sampler, whose kernels always run top_k, top_p, min_p and
// temperature once each, to the chains that are that one. A chain that differs in effect stays on the ordinary path,
// whichever way it differs; stages the config leaves neutral change nothing.
func TestStrataRoundSamplesOnlyItsOwnChain(t *testing.T) {
	t.Parallel()
	const (
		topK        = sampling.SamplerTopK
		topP        = sampling.SamplerTopP
		minP        = sampling.SamplerMinP
		temperature = sampling.SamplerTemperature
		xtc         = sampling.SamplerXTC
	)
	for _, test := range []struct {
		name   string
		config sampling.Config
		round  bool
	}{
		{"the default chain", sampling.Config{Temperature: 1, TopK: 20}, true},
		{"the native order", sampling.Config{Temperature: 1, TopK: 20, TopP: 0.9, MinP: 0.1, Samplers: []sampling.SamplerStage{topK, topP, minP, temperature}}, true},
		{"a neutral stage omitted", sampling.Config{Temperature: 1, TopK: 20, Samplers: []sampling.SamplerStage{topK, temperature}}, true},
		{"a neutral stage out of order", sampling.Config{Temperature: 1, TopK: 20, TopP: 1, Samplers: []sampling.SamplerStage{minP, xtc, topK, temperature, topP}}, true},
		{"an explicit empty chain", sampling.Config{Temperature: 1, TopK: 1, Samplers: []sampling.SamplerStage{}}, false},
		{"top_k left out", sampling.Config{Temperature: 1, TopK: 1, Samplers: []sampling.SamplerStage{temperature}}, false},
		{"an active top_p left out", sampling.Config{Temperature: 1, TopK: 20, TopP: 0.9, Samplers: []sampling.SamplerStage{topK, temperature}}, false},
		{"an active min_p left out", sampling.Config{Temperature: 1, TopK: 20, MinP: 0.1, Samplers: []sampling.SamplerStage{topK, temperature}}, false},
		{"temperature twice", sampling.Config{Temperature: 0.5, TopK: 20, Samplers: []sampling.SamplerStage{topK, temperature, temperature}}, false},
		{"temperature before top_k", sampling.Config{Temperature: 1, TopK: 20, Samplers: []sampling.SamplerStage{temperature, topK}}, false},
		{"top_p after min_p", sampling.Config{Temperature: 1, TopK: 20, TopP: 0.9, MinP: 0.1, Samplers: []sampling.SamplerStage{topK, minP, topP, temperature}}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			sampler, err := sampling.New(test.config)
			if err != nil {
				t.Fatal(err)
			}
			if _, round := strataSamplingOf(GenerateOptions{Sampler: sampler}); round != test.round {
				t.Fatalf("the round takes this chain: %t, want %t", round, test.round)
			}
		})
	}
}
