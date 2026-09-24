package tokenizer

import (
	"fmt"
	"strings"
	"testing"
)

func BenchmarkEncodeSharedBPE(b *testing.B) {
	vocab := mergeTestVocab(b)
	for _, units := range []int{8, 512} {
		text := strings.Repeat("abcde ", units)
		b.Run(fmt.Sprintf("bytes=%d", len(text)), func(b *testing.B) {
			ids, err := vocab.Encode(text, EncodeOptions{})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportMetric(float64(len(ids)), "tokens/op")
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := vocab.Encode(text, EncodeOptions{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
