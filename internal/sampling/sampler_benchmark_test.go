package sampling

import "testing"

const benchmarkVocabularySize = 4096

func BenchmarkSamplerPipeline(b *testing.B) {
	logits := make([]float32, benchmarkVocabularySize)
	for index := range logits {
		logits[index] = float32(index%97) / 97
	}
	sampler, err := New(Config{
		Seed: 1, Temperature: 0.8, TopK: 40, TopP: 0.95, MinP: 0.05,
	})
	if err != nil {
		b.Fatal(err)
	}
	if _, err := sampler.Sample(logits); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := sampler.Sample(logits); err != nil {
			b.Fatal(err)
		}
	}
}
