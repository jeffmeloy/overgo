package hfbpe

import (
	"slices"
	"strings"
	"testing"
)

func scanLegacyBPE(word string, ranks map[string]int) []string {
	parts := strings.Split(word, "")
	for len(parts) > 1 {
		best := -1
		var bestRank int
		for index := 0; index+1 < len(parts); index++ {
			if rank, ok := ranks[parts[index]+" "+parts[index+1]]; ok && (best < 0 || rank < bestRank) {
				best, bestRank = index, rank
			}
		}
		if best < 0 {
			break
		}
		parts[best] += parts[best+1]
		parts = append(parts[:best+1], parts[best+2:]...)
	}
	return parts
}

func TestSharedBPELegacyBoundaries(t *testing.T) {
	for _, test := range []struct {
		name, word string
		ranks      map[string]int
	}{
		{"empty", "", nil},
		{"single rune", "é", nil},
		{"no merge", "abc", nil},
		{"equal ranks", "abc", map[string]int{"a b": 4, "b c": 4}},
		{"integer ranks", "abc", map[string]int{"a b": 1<<24 + 1, "b c": 1 << 24}},
		{"unicode", "é😀é", map[string]int{"é 😀": 0, "😀 é": 1}},
		{"malformed bytes", "a\xff\xfeb", map[string]int{"a \xff": 0, "\xff \xfe": 1}},
	} {
		t.Run(test.name, func(t *testing.T) {
			tok := &Tokenizer{mergeRank: test.ranks}
			got := tok.bpe(test.word)
			want := scanLegacyBPE(test.word, test.ranks)
			if !slices.Equal(got, want) {
				t.Fatalf("word=%q pieces=%q want=%q", test.word, got, want)
			}
		})
	}
}
