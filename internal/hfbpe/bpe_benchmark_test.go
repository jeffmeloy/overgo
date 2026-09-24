package hfbpe

import (
	"fmt"
	"strings"
	"testing"
)

// benchmarkBPEEncoder keeps the vocabulary and declared no-regex ByteLevel
// splitting fixed across pre- and post-change complete Encode measurements.
func benchmarkBPEEncoder() *Tokenizer {
	tok := &Tokenizer{
		vocab:     make(map[string]int),
		mergeRank: make(map[string]int),
		special:   make(map[string]int),
	}
	tok.buildByteAlphabet()
	tok.preTokenize = func(text string) []string { return []string{text} }
	alphabet := []byte("abcde ")
	for _, value := range alphabet {
		symbol := string(tok.b2u[value])
		tok.vocab[symbol] = len(tok.vocab)
	}
	for _, left := range alphabet {
		for _, right := range alphabet {
			a := string(tok.b2u[left])
			b := string(tok.b2u[right])
			tok.mergeRank[a+" "+b] = len(tok.mergeRank)
			tok.vocab[a+b] = len(tok.vocab)
		}
	}
	return tok
}

func BenchmarkEncodeSharedBPE(b *testing.B) {
	tok := benchmarkBPEEncoder()
	for _, units := range []int{8, 512} {
		text := strings.Repeat("abcde ", units)
		b.Run(fmt.Sprintf("bytes=%d", len(text)), func(b *testing.B) {
			ids, err := tok.Encode(text)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportMetric(float64(len(ids)), "tokens/op")
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := tok.Encode(text); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
