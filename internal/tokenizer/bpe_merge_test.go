package tokenizer

import (
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gguf"
)

// applyBPEByScan is the pair-scan merge the heap merge replaced: every
// step scans all adjacent pairs for the lowest rank, leftmost first. It
// is the reference the heap must reproduce symbol for symbol.
func (v *Vocab) applyBPEByScan(word string) []string {
	symbols := make([]string, 0, len(word))
	for _, symbol := range word {
		symbols = append(symbols, string(symbol))
	}
	for len(symbols) > 1 {
		bestIndex := -1
		bestRank := int(^uint(0) >> 1)
		for i := 0; i+1 < len(symbols); i++ {
			rank, ok := v.mergeRank[pair{left: symbols[i], right: symbols[i+1]}]
			if ok && rank < bestRank {
				bestIndex = i
				bestRank = rank
			}
		}
		if bestIndex < 0 {
			break
		}
		symbols[bestIndex] += symbols[bestIndex+1]
		copy(symbols[bestIndex+1:], symbols[bestIndex+2:])
		symbols = symbols[:len(symbols)-1]
	}
	return symbols
}

// mergeTestVocab builds a Gemma-4-style vocabulary over a small alphabet
// whose merge table has every two-letter pair, then three- and
// four-letter pieces, in a shuffled rank order, so merges compete and
// rank ties fall to the leftmost pair.
func mergeTestVocab(t testing.TB) *Vocab {
	t.Helper()
	alphabet := []string{"a", "b", "c", "d", "e", "▁"}
	var pieces []string
	var merges []string
	for _, left := range alphabet {
		for _, right := range alphabet {
			pieces = append(pieces, left+right)
			merges = append(merges, left+" "+right)
		}
	}
	for _, left := range alphabet {
		for _, right := range []string{"ab", "ba", "cd", "▁a", "e▁"} {
			pieces = append(pieces, left+right)
			merges = append(merges, left+" "+right)
		}
	}
	for _, left := range []string{"ab", "cd", "▁a"} {
		for _, right := range []string{"ab", "ba", "cd", "▁a", "e▁"} {
			pieces = append(pieces, left+right)
			merges = append(merges, left+" "+right)
		}
	}
	rng := rand.New(rand.NewPCG(7, 11))
	rng.Shuffle(len(merges), func(i, j int) { merges[i], merges[j] = merges[j], merges[i] })
	tokens := append([]string{"<bos>", "\n"}, alphabet...)
	tokens = append(tokens, pieces...)
	types := make([]int32, len(tokens))
	for index := range types {
		types[index] = 1
	}
	types[0] = 3
	file := &gguf.File{Metadata: []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "gemma4"),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		array("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
		array("tokenizer.ggml.merges", gguf.ValueTypeString, merges),
		scalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(0)),
		scalar("tokenizer.ggml.add_bos_token", gguf.ValueTypeBool, true),
	}}
	vocab, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	return vocab
}

// TestApplyBPEHeapMatchesPairScan: the heap merge reproduces the pair
// scan on random lines of every length, including lines a single symbol
// long and lines with no mergeable pair.
func TestApplyBPEHeapMatchesPairScan(t *testing.T) {
	vocab := mergeTestVocab(t)
	rng := rand.New(rand.NewPCG(3, 5))
	alphabet := []rune("abcde▁")
	for trial := range 400 {
		length := trial % 64
		if trial%7 == 0 {
			length = 500 + rng.IntN(400)
		}
		var word strings.Builder
		for range length {
			word.WriteRune(alphabet[rng.IntN(len(alphabet))])
		}
		got := vocab.applyBPE(word.String())
		want := vocab.applyBPEByScan(word.String())
		if !slices.Equal(got, want) {
			t.Fatalf("trial %d word %q: heap merge %q, pair scan %q", trial, word.String(), got, want)
		}
	}
}

// longMergeLine is a 12,000-character line, the size of a MuSR
// narrative under the Gemma 4 newline split.
func longMergeLine() string {
	rng := rand.New(rand.NewPCG(9, 13))
	alphabet := []rune("abcde▁")
	var word strings.Builder
	for range 12000 {
		word.WriteRune(alphabet[rng.IntN(len(alphabet))])
	}
	return word.String()
}

// TestApplyBPEHeapMatchesPairScanOnALongLine: the parity holds on the
// narrative-length line as well, where the pair scan is at its
// slowest.
func TestApplyBPEHeapMatchesPairScanOnALongLine(t *testing.T) {
	vocab := mergeTestVocab(t)
	line := longMergeLine()
	got := vocab.applyBPE(line)
	if want := vocab.applyBPEByScan(line); !slices.Equal(got, want) {
		t.Fatalf("heap merge of the long line differs from the pair scan: %d vs %d pieces", len(got), len(want))
	}
}

// BenchmarkApplyBPELongLine measures the heap merge on the
// narrative-length line; the pair scan it replaced took about two
// seconds on the same input.
func BenchmarkApplyBPELongLine(b *testing.B) {
	vocab := mergeTestVocab(b)
	line := longMergeLine()
	b.ReportAllocs()
	for b.Loop() {
		vocab.applyBPE(line)
	}
}
