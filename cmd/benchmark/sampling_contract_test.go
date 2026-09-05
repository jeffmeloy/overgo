package main

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/sampling"
)

func TestBenchmarkSamplingContract(t *testing.T) {
	t.Run("greedy budget preserves the EOG winner", func(t *testing.T) {
		generation, err := benchmarkGenerationOptions(options{Tokens: 4, Temperature: 0, TopK: 40})
		if err != nil {
			t.Fatal(err)
		}
		sampler := generation.Sampler
		if !generation.ContinueAfterEOG || generation.MaxNewTokens != 4 || !generation.DeviceGreedy || !sampler.IsRawGreedy() {
			t.Error("greedy benchmark cannot use device argmax")
		}
		id, err := sampler.Sample([]float32{10, 1, 0})
		if err != nil || id != 0 {
			t.Fatalf("greedy winner changed: id=%d error=%v", id, err)
		}
	})
	t.Run("explicit top-k is applied", func(t *testing.T) {
		sampler, err := benchmarkSampler(options{Temperature: 0.5, TopK: 1})
		if err != nil {
			t.Fatal(err)
		}
		logits := []float32{1, 0.9, 0.8}
		for range len(logits) {
			id, err := sampler.Sample(logits)
			if err != nil || id != 0 {
				t.Fatalf("top-k=1 selected a lower logit: id=%d error=%v", id, err)
			}
		}
	})
	t.Run("historical banned winner is a different protocol", func(t *testing.T) {
		legacy, err := sampling.New(sampling.Config{Temperature: 0, TopK: 40,
			LogitBiases: []sampling.LogitBias{{Token: 0, Bias: sampling.BannedLogit()}}})
		if err != nil {
			t.Fatal(err)
		}
		id, err := legacy.Sample([]float32{10, 1, 0})
		if err != nil || id == 0 || legacy.IsRawGreedy() {
			t.Fatalf("legacy negative control changed: %d %v", id, err)
		}
	})
	t.Run("temperature scales probabilities and bounded top-k agrees", func(t *testing.T) {
		sampler, err := benchmarkSampler(options{Temperature: 0.5, TopK: 2})
		if err != nil {
			t.Fatal(err)
		}
		probabilities, err := sampler.SampleWithHistoryProbabilities([]float32{1, 0, -1}, nil, 3)
		if err != nil {
			t.Fatal(err)
		}
		want := 1 / (1 + math.Exp(-2))
		if len(probabilities.Top) != 2 || probabilities.Top[0].ID != 0 || math.Abs(probabilities.Top[0].Probability-want) > 1e-12 {
			t.Fatalf("top-k/temperature probabilities = %+v, want highest=%g", probabilities.Top, want)
		}
		full, _ := benchmarkSampler(options{Temperature: 0.5, TopK: 2})
		bounded, _ := benchmarkSampler(options{Temperature: 0.5, TopK: 2})
		logits := []float32{1, 0, -1}
		for range len(logits) {
			a, err := full.Sample(logits)
			if err != nil {
				t.Fatal(err)
			}
			b, err := bounded.SampleTopK([]int{0, 1}, logits[:2], len(logits))
			if err != nil || a != b {
				t.Fatalf("bounded selection %d != full %d: %v", b, a, err)
			}
		}
		generation, err := benchmarkGenerationOptions(options{Temperature: 0.5, TopK: 2, Tokens: 4})
		if err != nil || generation.DeviceGreedy || !generation.ContinueAfterEOG {
			t.Fatalf("stochastic generation options = %+v %v", generation, err)
		}
	})
	t.Run("publication refuses unbound or incomplete protocols", func(t *testing.T) {
		valid := benchmarkResult{EndOfSequenceIgnored: true, SamplingProtocol: greedyBudgetProtocol, TokensPerSequence: 4, RequestedRuns: 1,
			Runs: []runMetrics{{OutputTokens: 4, HostLogitTokens: 1, DeviceSelectedTokens: 3, PromptTokens: 2, TotalMilliseconds: 10}}}
		if err := validateBenchmarkResult(valid); err != nil {
			t.Fatal(err)
		}
		for name, mutate := range map[string]func(*benchmarkResult){
			"legacy unknown protocol":   func(r *benchmarkResult) { r.SamplingProtocol = "" },
			"different protocol":        func(r *benchmarkResult) { r.SamplingProtocol = topKBudgetProtocol },
			"undeclared stages":         func(r *benchmarkResult) { r.SamplerOrder = []sampling.SamplerStage{sampling.SamplerTopK} },
			"incomplete output":         func(r *benchmarkResult) { r.Runs[0].OutputTokens-- },
			"unobserved selection path": func(r *benchmarkResult) { r.Runs[0].DeviceSelectedTokens = 0 },
			"unrequested top-k path":    func(r *benchmarkResult) { r.Runs[0].DeviceSelectedTokens--; r.Runs[0].DeviceTopKTokens++ },
			"incomplete repeats":        func(r *benchmarkResult) { r.RequestedRuns++ },
			"EOG stopping":              func(r *benchmarkResult) { r.EndOfSequenceIgnored = false },
			"invalid temperature":       func(r *benchmarkResult) { r.Temperature = math.NaN() },
			"false batch budget":        func(r *benchmarkResult) { r.BatchSequences = 2 },
		} {
			t.Run(name, func(t *testing.T) {
				changed := valid
				changed.Runs = slices.Clone(valid.Runs)
				mutate(&changed)
				if _, _, _, err := benchmarkClaim(changed, "0123456789abcdef0123456789abcdef01234567"); err == nil {
					t.Fatal("invalid claim accepted")
				}
			})
		}
	})
}
