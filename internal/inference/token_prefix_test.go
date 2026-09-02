package inference

import (
	"testing"

	"overgo/internal/tokenizer"
)

func TestCommonTokenPrefixStopsAtTheFirstMergedToken(t *testing.T) {
	prompt := []tokenizer.TokenID{1, 2, 3}
	cases := []struct {
		name string
		full []tokenizer.TokenID
		want int
	}{
		{name: "prompt kept as prefix", full: []tokenizer.TokenID{1, 2, 3, 9}, want: 3},
		{name: "last prompt token merged into the candidate", full: []tokenizer.TokenID{1, 2, 7, 9}, want: 2},
		{name: "no shared context", full: []tokenizer.TokenID{5}, want: 0},
		{name: "shorter than the prompt", full: []tokenizer.TokenID{1, 2}, want: 2},
	}
	for _, tc := range cases {
		if got := commonTokenPrefix(prompt, tc.full); got != tc.want {
			t.Errorf("%s: common prefix = %d, want %d", tc.name, got, tc.want)
		}
	}
}
