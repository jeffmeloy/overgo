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
