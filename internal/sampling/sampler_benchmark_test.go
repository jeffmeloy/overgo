package sampling

import "testing"

const benchmarkVocabularySize = 4096

func BenchmarkSamplerPipeline(b *testing.B) {
	logits := make([]float32, benchmarkVocabularySize)
	history := make([]int, benchmarkVocabularySize/4)
	for index := range logits {
		logits[index] = float32(index%97) / 97
	}
	for index := range history {
		history[index] = index % benchmarkVocabularySize
	}
	sampler, err := New(Config{
		Seed: 1, Temperature: 0.8, TopK: 40, TopP: 0.95, MinP: 0.05,
		RepeatLastN: -1, RepeatPenalty: 1.1,
		DryMultiplier: 1, DryBase: 2, DryAllowedLength: 2, DryPenaltyLastN: -1,
	})
	if err != nil {
		b.Fatal(err)
	}
	if _, err := sampler.SampleWithHistory(logits, history); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := sampler.SampleWithHistory(logits, history); err != nil {
			b.Fatal(err)
		}
	}
}
