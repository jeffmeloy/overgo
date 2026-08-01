package sampling

import (
	"bytes"
	"math"
	"testing"
)

func TestSpeculativeSampleMatchesCommittedSamplerState(t *testing.T) {
	tests := []struct {
		name   string
		config Config
	}{
		{name: "chain", config: Config{
			Temperature: 0.8, TopK: 3, TopP: 1, Seed: 17,
		}},
		{name: "adaptive", config: Config{
			Temperature: 1, AdaptiveTarget: 0.2, AdaptiveDecay: 0.9,
			Samplers: []SamplerStage{SamplerAdaptiveP}, Seed: 23,
		}},
		{name: "mirostat-v1", config: Config{
			Temperature: 1, Mirostat: 1, MirostatTau: 5, MirostatEta: 0.1, Seed: 29,
		}},
		{name: "mirostat-v2", config: Config{
			Temperature: 1, Mirostat: 2, MirostatTau: 5, MirostatEta: 0.1, Seed: 31,
		}},
	}
	logits := []float32{0.2, 1.7, -0.4, 0.9}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sampler, err := New(test.config)
			if err != nil {
				t.Fatal(err)
			}
			probe, err := sampler.SampleWithHistoryProbabilities(logits, []int{1}, len(logits))
			if err != nil {
				t.Fatal(err)
			}
			sampler.Reset()
			selected, err := sampler.SampleWithHistory(logits, []int{1})
			if err != nil {
				t.Fatal(err)
			}
			baseline, err := sampler.SaveState()
			if err != nil {
				t.Fatal(err)
			}
			sampler.Reset()
			result, err := sampler.SpeculativeSample(logits, []int{1}, probe.Top, selected)
			if err != nil {
				t.Fatal(err)
			}
			if !result.Accepted || result.Token != selected {
				t.Fatalf("result = %+v, selected = %d", result, selected)
			}
			state, err := sampler.SaveState()
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(state, baseline) {
				t.Fatal("speculative commit diverged from ordinary sampling state")
			}
		})
	}
}

func TestSpeculativeSampleUsesResidualCorrection(t *testing.T) {
	sampler, err := New(Config{
		Temperature: 1, TopK: 1, TopP: 1, Seed: 7,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := sampler.SpeculativeSample(
		[]float32{0, 10}, nil,
		[]TokenProbability{{ID: 0, Probability: 0.75}, {ID: 1, Probability: 0.25}},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Accepted || result.Token != 1 || result.TargetProbability != 0 ||
		result.DraftProbability != 0.75 || len(result.Target) != 1 ||
		result.Target[0] != (TokenProbability{ID: 1, Probability: 1}) {
		t.Fatalf("result = %+v", result)
	}
}

func TestSpeculativeSampleRatioOracle(t *testing.T) {
	sampler, err := New(Config{Temperature: 1, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	result, err := sampler.SpeculativeSample(
		[]float32{float32(math.Log(0.25)), float32(math.Log(0.75))}, nil,
		[]TokenProbability{{ID: 0, Probability: 0.5}, {ID: 1, Probability: 0.5}},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(result.TargetProbability-0.25) > 1e-7 || result.DraftProbability != 0.5 {
		t.Fatalf("result = %+v", result)
	}
	if result.Accepted || result.Token != 1 {
		t.Fatalf("seeded ratio oracle = %+v", result)
	}
}

func TestSpeculativeSampleRejectsInvalidDraft(t *testing.T) {
	sampler, err := New(Config{Temperature: 1})
	if err != nil {
		t.Fatal(err)
	}
	invalid := [][]TokenProbability{
		nil,
		{{ID: 0, Probability: 0.4}},
		{{ID: 0, Probability: 0.5}, {ID: 0, Probability: 0.5}},
		{{ID: 0, Probability: math.NaN()}, {ID: 1, Probability: 1}},
	}
	for index, draft := range invalid {
		if _, err := sampler.SpeculativeSample([]float32{0, 1}, nil, draft, 0); err == nil {
			t.Fatalf("invalid draft %d accepted", index)
		}
	}
}
