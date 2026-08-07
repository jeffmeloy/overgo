package sampling

import (
	"math"
	"testing"
)

func TestGreedyAndTieBreak(t *testing.T) {
	sampler, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	id, err := sampler.Sample([]float32{1, 3, 3, 2})
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("greedy ID = %d, want 1", id)
	}
}

func TestIsRawGreedyRejectsLogitTransforms(t *testing.T) {
	greedy, err := New(Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if !greedy.IsRawGreedy() {
		t.Fatal("plain greedy sampler was not recognized")
	}
	for name, config := range map[string]Config{
		"temperature":         {Temperature: 0.5},
		"dynamic temperature": {Temperature: 0, DynatempRange: 0.5},
		"no greedy stage":     {Temperature: 0, Samplers: []SamplerStage{SamplerTopK}},
		"penalty":             {Temperature: 0, RepeatLastN: -1, RepeatPenalty: 1.1},
		"bias":                {Temperature: 0, LogitBiases: []LogitBias{{Token: 1, Bias: 1}}},
		"xtc":                 {Temperature: 0, XTCProbability: 1},
	} {
		t.Run(name, func(t *testing.T) {
			sampler, newErr := New(config)
			if newErr != nil {
				t.Fatal(newErr)
			}
			if sampler.IsRawGreedy() {
				t.Fatal("transforming sampler was recognized as raw greedy")
			}
		})
	}
}

func TestTopKOneIsGreedy(t *testing.T) {
	sampler, err := New(Config{Temperature: 1, TopK: 1, TopP: 1, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		id, sampleErr := sampler.Sample([]float32{1, 5, 3})
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if id != 1 {
			t.Fatalf("top-k=1 sampled %d, want 1", id)
		}
	}
}

func TestBoundedTopKMatchesFullPipeline(t *testing.T) {
	config := Config{
		Temperature: 0.8, TopK: 3, TopP: 0.9, MinP: 0.05, Seed: 42,
	}
	full, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	bounded, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	if limit, ok := bounded.BoundedTopK(); !ok || limit != config.TopK {
		t.Fatalf("bounded top-K = %d, %v", limit, ok)
	}
	logits := []float32{0.2, 1.4, -0.1, 2.1, 0.7}
	ids := []int{3, 1, 4}
	values := []float32{logits[3], logits[1], logits[4]}
	for index := range 50 {
		want, sampleErr := full.Sample(logits)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		got, sampleErr := bounded.SampleTopK(ids, values, len(logits))
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if got != want {
			t.Fatalf("sample %d = %d, want %d", index, got, want)
		}
	}
}

func TestBoundedTopKRejectsPrefixTransforms(t *testing.T) {
	for name, config := range map[string]Config{
		"penalty": {Temperature: 0.8, TopK: 3, RepeatPenalty: 1.1},
		"bias": {
			Temperature: 0.8, TopK: 3,
			LogitBiases: []LogitBias{{Token: 4, Bias: 1}},
		},
		"sigma": {Temperature: 0.8, TopK: 3, TopNSigma: 1},
		"ordered filter": {
			Temperature: 0.8, TopK: 3, TopP: 0.9,
			Samplers: []SamplerStage{SamplerTopP, SamplerTopK, SamplerTemperature},
		},
	} {
		t.Run(name, func(t *testing.T) {
			sampler, err := New(config)
			if err != nil {
				t.Fatal(err)
			}
			if limit, ok := sampler.BoundedTopK(); ok || limit != 0 {
				t.Fatalf("unsafe bounded top-K = %d, %v", limit, ok)
			}
		})
	}
}

func TestSamplingIsSeedDeterministic(t *testing.T) {
	first, _ := New(Config{Temperature: 0.8, TopK: 3, TopP: 0.9, Seed: 42})
	second, _ := New(Config{Temperature: 0.8, TopK: 3, TopP: 0.9, Seed: 42})
	logits := []float32{1, 2, 3, 4, 5}
	for index := range 100 {
		a, err := first.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
		b, err := second.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
		if a != b {
			t.Fatalf("sample %d differs: %d and %d", index, a, b)
		}
	}
}

func TestExtendedSamplingIsSeedDeterministic(t *testing.T) {
	config := Config{
		Temperature:      0.8,
		TopK:             4,
		TopP:             0.9,
		MinP:             0.05,
		TypicalP:         0.95,
		RepeatLastN:      -1,
		RepeatPenalty:    1.1,
		PresencePenalty:  0.2,
		FrequencyPenalty: 0.1,
		Seed:             42,
	}
	first, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	second, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{1, 2, 3, 4, 5}
	history := []int{4, 4, 3}
	for index := range 100 {
		a, sampleErr := first.SampleWithHistory(logits, history)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		b, sampleErr := second.SampleWithHistory(logits, history)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if a != b {
			t.Fatalf("extended sample %d differs: %d and %d", index, a, b)
		}
		history = append(history, a)
	}
}

func TestTopPRestrictsTail(t *testing.T) {
	sampler, err := New(Config{Temperature: 0.1, TopP: 0.5, Seed: 9})
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		id, sampleErr := sampler.Sample([]float32{10, 1, 0})
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if id != 0 {
			t.Fatalf("top-p sampled tail ID %d", id)
		}
	}
}

func TestMinPRestrictsRelativeTail(t *testing.T) {
	sampler, err := New(Config{
		Temperature: 1,
		TopP:        1,
		MinP:        0.5,
		Seed:        9,
	})
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		id, sampleErr := sampler.Sample([]float32{0, -1, -2})
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if id != 0 {
			t.Fatalf("min-p sampled tail ID %d", id)
		}
	}
}

func TestTypicalPRestrictsAtypicalTail(t *testing.T) {
	sampler, err := New(Config{
		Temperature: 1,
		TopP:        1,
		TypicalP:    0.5,
		Seed:        3,
	})
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		id, sampleErr := sampler.Sample([]float32{3, 0, 0})
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if id != 0 {
			t.Fatalf("typical-p sampled tail ID %d", id)
		}
	}
}

func TestRepeatPenaltyChangesGreedyChoice(t *testing.T) {
	sampler, err := New(Config{RepeatLastN: -1, RepeatPenalty: 2})
	if err != nil {
		t.Fatal(err)
	}
	id, err := sampler.SampleWithHistory([]float32{10, 9}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("penalized greedy ID = %d, want 1", id)
	}
}

func TestPresenceAndFrequencyPenalties(t *testing.T) {
	sampler, err := New(Config{
		RepeatLastN:      -1,
		PresencePenalty:  1,
		FrequencyPenalty: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	id, err := sampler.SampleWithHistory([]float32{10, 7}, []int{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("presence/frequency penalized ID = %d, want 1", id)
	}
}

func TestRepeatWindowUsesOnlySuffix(t *testing.T) {
	sampler, err := New(Config{RepeatLastN: 1, RepeatPenalty: 2})
	if err != nil {
		t.Fatal(err)
	}
	id, err := sampler.SampleWithHistory([]float32{10, 9}, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if id != 0 {
		t.Fatalf("suffix-window greedy ID = %d, want 0", id)
	}
}

func TestSamplerStageOrderChangesCandidateSet(t *testing.T) {
	base := Config{
		RepeatLastN:   -1,
		RepeatPenalty: 10,
		TopK:          1,
	}
	penaltiesFirst := base
	penaltiesFirst.Samplers = []SamplerStage{
		SamplerPenalties,
		SamplerTopK,
		SamplerTemperature,
	}
	topKFirst := base
	topKFirst.Samplers = []SamplerStage{
		SamplerTopK,
		SamplerPenalties,
		SamplerTemperature,
	}
	first, err := New(penaltiesFirst)
	if err != nil {
		t.Fatal(err)
	}
	second, err := New(topKFirst)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{3, 2, 1}
	history := []int{0}
	if id, sampleErr := first.SampleWithHistory(logits, history); sampleErr != nil || id != 1 {
		t.Fatalf("penalties-first ID = %d, %v; want 1", id, sampleErr)
	}
	if id, sampleErr := second.SampleWithHistory(logits, history); sampleErr != nil || id != 0 {
		t.Fatalf("top-k-first ID = %d, %v; want 0", id, sampleErr)
	}
}

func TestParseSamplerOrderAndRejectUnknown(t *testing.T) {
	order, err := ParseSamplerOrder("top_k;penalties;top_k;temperature")
	if err != nil {
		t.Fatal(err)
	}
	want := []SamplerStage{
		SamplerTopK,
		SamplerPenalties,
		SamplerTopK,
		SamplerTemperature,
	}
	if len(order) != len(want) {
		t.Fatalf("order = %v, want %v", order, want)
	}
	for index := range want {
		if order[index] != want[index] {
			t.Fatalf("order = %v, want %v", order, want)
		}
	}
	empty, err := ParseSamplerOrder("none")
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("empty order = %#v, %v", empty, err)
	}
	if _, err := ParseSamplerOrder("top_k;tfs_z"); err == nil {
		t.Fatal("unsupported sampler was accepted")
	}
	if _, err := New(Config{Samplers: []SamplerStage{"unknown"}}); err == nil {
		t.Fatal("unknown configured sampler was accepted")
	}
}

func TestTopNSigmaMatchesPinnedUpstreamVectors(t *testing.T) {
	cases := []struct {
		n    float32
		want []float64
	}{
		{n: 1, want: []float64{0, 0, 0.428571, 0.571429}},
		{n: 0, want: []float64{0.1, 0.2, 0.3, 0.4}},
		{n: 3, want: []float64{0.1, 0.2, 0.3, 0.4}},
	}
	for _, test := range cases {
		sampler, err := New(Config{TopNSigma: test.n})
		if err != nil {
			t.Fatal(err)
		}
		candidates := probabilityCandidates([]float64{0.1, 0.2, 0.3, 0.4})
		candidates, err = sampler.applySamplerStage(
			candidates,
			len(candidates),
			nil,
			SamplerTopNSigma,
		)
		if err != nil {
			t.Fatal(err)
		}
		got := normalizedCandidateProbabilities(t, candidates, 4)
		for index := range test.want {
			if math.Abs(got[index]-test.want[index]) > 1e-5 {
				t.Fatalf("top-n-sigma %v probabilities = %v, want %v", test.n, got, test.want)
			}
		}
	}
}

func TestXTCMatchesPinnedUpstreamVectors(t *testing.T) {
	cases := []struct {
		threshold float32
		want      []float64
	}{
		{threshold: 0.09, want: []float64{0, 0, 0, 1}},
		{threshold: 0.19, want: []float64{0, 0, 2.0 / 3.0, 1.0 / 3.0}},
		{threshold: 0.29, want: []float64{0, 0.5, 1.0 / 3.0, 1.0 / 6.0}},
		{threshold: 0.39, want: []float64{0.4, 0.3, 0.2, 0.1}},
	}
	for _, test := range cases {
		sampler, err := New(Config{
			XTCProbability: 1,
			XTCThreshold:   test.threshold,
		})
		if err != nil {
			t.Fatal(err)
		}
		candidates := probabilityCandidates([]float64{0.4, 0.3, 0.2, 0.1})
		candidates, err = sampler.applySamplerStage(
			candidates,
			len(candidates),
			nil,
			SamplerXTC,
		)
		if err != nil {
			t.Fatal(err)
		}
		got := normalizedCandidateProbabilities(t, candidates, 4)
		for index := range test.want {
			if math.Abs(got[index]-test.want[index]) > 1e-5 {
				t.Fatalf("XTC threshold %v probabilities = %v, want %v", test.threshold, got, test.want)
			}
		}
	}
}

func TestMinKeepRetainsProbabilityFilterFloor(t *testing.T) {
	sampler, err := New(Config{TopP: 0.01, MinP: 0.9, MinKeep: 2})
	if err != nil {
		t.Fatal(err)
	}
	candidates := probabilityCandidates([]float64{0.7, 0.2, 0.09, 0.01})
	for _, stage := range []SamplerStage{SamplerTopP, SamplerMinP} {
		candidates, err = sampler.applySamplerStage(candidates, 4, nil, stage)
		if err != nil {
			t.Fatal(err)
		}
		if len(candidates) != 2 {
			t.Fatalf("%s retained %d candidates, want 2", stage, len(candidates))
		}
		candidates = probabilityCandidates([]float64{0.7, 0.2, 0.09, 0.01})
	}
}

func TestDynamicTemperatureMatchesPinnedFormula(t *testing.T) {
	sampler, err := New(Config{
		Temperature:      0.8,
		DynatempRange:    0.4,
		DynatempExponent: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	candidates := probabilityCandidates([]float64{0.1, 0.2, 0.3, 0.4})
	candidates, err = sampler.applySamplerStage(
		candidates,
		len(candidates),
		nil,
		SamplerTemperature,
	)
	if err != nil {
		t.Fatal(err)
	}
	got := normalizedCandidateProbabilities(t, candidates, 4)
	want := []float64{
		0.10798992400900212,
		0.20494320945284927,
		0.2981257733178854,
		0.38894109322026316,
	}
	for index := range want {
		if math.Abs(got[index]-want[index]) > 1e-6 {
			t.Fatalf("dynamic-temperature probabilities = %v, want %v", got, want)
		}
	}
}

func TestZeroDynamicRangeUsesFixedTemperature(t *testing.T) {
	sampler, err := New(Config{
		Temperature:      1,
		DynatempExponent: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	candidates := probabilityCandidates([]float64{0.1, 0.2, 0.3, 0.4})
	candidates, err = sampler.applySamplerStage(
		candidates,
		len(candidates),
		nil,
		SamplerTemperature,
	)
	if err != nil {
		t.Fatal(err)
	}
	got := normalizedCandidateProbabilities(t, candidates, 4)
	want := []float64{0.1, 0.2, 0.3, 0.4}
	for index := range want {
		if math.Abs(got[index]-want[index]) > 1e-6 {
			t.Fatalf("fixed-temperature probabilities = %v, want %v", got, want)
		}
	}
}

func TestAdaptivePFavorsTargetProbability(t *testing.T) {
	const runs = 100
	selectedTarget := 0
	for seed := range runs {
		sampler, err := New(Config{
			AdaptiveTarget: 0.25,
			AdaptiveDecay:  0.9,
			Seed:           int64(seed),
			Samplers:       []SamplerStage{SamplerAdaptiveP},
		})
		if err != nil {
			t.Fatal(err)
		}
		id, err := sampler.Sample([]float32{
			float32(math.Log(0.6)),
			float32(math.Log(0.25)),
			float32(math.Log(0.1)),
			float32(math.Log(0.05)),
		})
		if err != nil {
			t.Fatal(err)
		}
		if id == 1 {
			selectedTarget++
		}
	}
	if selectedTarget < 70 {
		t.Fatalf("adaptive-p selected target-near token %d/%d times", selectedTarget, runs)
	}
}

func TestAdaptivePIsTerminalRegardlessOfListedPosition(t *testing.T) {
	sampler, err := New(Config{
		TopK:           1,
		AdaptiveTarget: 0.25,
		AdaptiveDecay:  0.9,
		Samplers: []SamplerStage{
			SamplerAdaptiveP,
			SamplerTopK,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range 10 {
		id, err := sampler.Sample([]float32{4, 3, 2})
		if err != nil {
			t.Fatal(err)
		}
		if id != 0 {
			t.Fatalf("terminal adaptive-p bypassed later top-k: ID %d", id)
		}
	}
}

func TestLogitBiasAdjustsAndBansTokens(t *testing.T) {
	sampler, err := New(Config{
		LogitBiases: []LogitBias{
			{Token: 1, Bias: 3},
			{Token: 2, Bias: float32(math.Inf(-1))},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	id, err := sampler.Sample([]float32{2, 0, 100})
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("biased greedy ID = %d, want 1", id)
	}
	config := sampler.Config()
	config.LogitBiases[0].Bias = -100
	if sampler.Config().LogitBiases[0].Bias != 3 {
		t.Fatal("sampler logit biases are mutable through Config")
	}
	if _, err := sampler.Sample([]float32{1, 2}); err == nil {
		t.Fatal("out-of-range configured logit bias was accepted")
	}
}

func probabilityCandidates(probabilities []float64) []candidate {
	result := make([]candidate, len(probabilities))
	for index, probability := range probabilities {
		result[index] = candidate{id: index, scaledLogit: math.Log(probability)}
	}
	return result
}

func normalizedCandidateProbabilities(
	t *testing.T,
	candidates []candidate,
	vocabularySize int,
) []float64 {
	t.Helper()
	total, err := candidateProbabilities(candidates)
	if err != nil {
		t.Fatal(err)
	}
	result := make([]float64, vocabularySize)
	for _, item := range candidates {
		result[item.id] = item.probability / total
	}
	return result
}

func TestSamplerRejectsInvalidExtendedConfig(t *testing.T) {
	cases := []Config{
		{MinP: 1.1},
		{TypicalP: 1.1},
		{DynatempRange: -1},
		{DynatempRange: float32(math.Inf(1))},
		{DynatempExponent: float32(math.NaN())},
		{AdaptiveTarget: 1.1},
		{AdaptiveTarget: float32(math.Inf(1))},
		{AdaptiveDecay: -0.1},
		{AdaptiveDecay: 1},
		{LogitBiases: []LogitBias{{Token: -1}}},
		{LogitBiases: []LogitBias{{Token: 1, Bias: float32(math.NaN())}}},
		{LogitBiases: []LogitBias{{Token: 1, Bias: float32(math.Inf(1))}}},
		{TopNSigma: float32(math.NaN())},
		{XTCProbability: -1},
		{XTCProbability: 1.1},
		{XTCThreshold: -1},
		{XTCThreshold: 1.1},
		{MinKeep: -1},
		{RepeatLastN: -2},
		{RepeatPenalty: -1},
		{DryMultiplier: -1},
		{DryBase: 0.5},
		{DryAllowedLength: -1},
		{DryPenaltyLastN: -2},
		{DryBreakers: [][]int{{}}},
		{DryBreakers: [][]int{{-1}}},
		{Mirostat: 3},
		{Mirostat: 2, MirostatTau: -1},
		{Mirostat: 2, MirostatEta: -1},
	}
	for _, config := range cases {
		if _, err := New(config); err == nil {
			t.Fatalf("New(%+v) succeeded, want error", config)
		}
	}
}

func TestMirostatV1IsDeterministicAndAdaptive(t *testing.T) {
	config := Config{
		Temperature: 1,
		Mirostat:    1,
		MirostatTau: 3,
		MirostatEta: 0.2,
		Seed:        23,
	}
	first, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	second, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, 64)
	for index := range logits {
		logits[index] = float32(-1.5 * math.Log(float64(index+1)))
	}
	initialMu := first.mu
	for index := range 50 {
		a, sampleErr := first.Sample(logits)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		b, sampleErr := second.Sample(logits)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if a != b || first.mu != second.mu {
			t.Fatalf("Mirostat v1 step %d differs: IDs %d/%d mu %v/%v", index, a, b, first.mu, second.mu)
		}
	}
	if first.mu == initialMu {
		t.Fatal("Mirostat v1 mu did not adapt")
	}
}

func TestMirostatV2IsDeterministicAndAdaptive(t *testing.T) {
	config := Config{
		Temperature: 1,
		Mirostat:    2,
		MirostatTau: 2,
		MirostatEta: 0.5,
		Seed:        17,
	}
	first, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	second, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{5, 4, 3, 2}
	initialMu := first.mu
	for index := range 50 {
		a, sampleErr := first.Sample(logits)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		b, sampleErr := second.Sample(logits)
		if sampleErr != nil {
			t.Fatal(sampleErr)
		}
		if a != b || first.mu != second.mu {
			t.Fatalf("Mirostat step %d differs: IDs %d/%d mu %v/%v", index, a, b, first.mu, second.mu)
		}
	}
	if first.mu == initialMu {
		t.Fatal("Mirostat mu did not adapt")
	}
}

func TestMirostatResetRestoresState(t *testing.T) {
	sampler, err := New(Config{
		Temperature: 1,
		Mirostat:    2,
		MirostatTau: 2,
		MirostatEta: 0.5,
		Seed:        9,
	})
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{3, 2, 1}
	first, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	sampler.Reset()
	if sampler.mu != 4 {
		t.Fatalf("reset mu = %v, want 4", sampler.mu)
	}
	repeated, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if repeated != first {
		t.Fatalf("reset sample = %d, want first sample %d", repeated, first)
	}
}

func TestDRYPenalizesExtendedSequence(t *testing.T) {
	sampler, err := New(Config{
		DryMultiplier:    1,
		DryBase:          2,
		DryAllowedLength: 2,
		DryPenaltyLastN:  -1,
	})
	if err != nil {
		t.Fatal(err)
	}
	id, err := sampler.SampleWithHistory(
		[]float32{10, 9, 0},
		[]int{0, 1, 2, 0, 1, 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("DRY greedy ID = %d, want 1", id)
	}
}

func TestDRYDisabledLeavesGreedyChoice(t *testing.T) {
	sampler, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	id, err := sampler.SampleWithHistory(
		[]float32{10, 9, 0},
		[]int{0, 1, 2, 0, 1, 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	if id != 0 {
		t.Fatalf("disabled DRY greedy ID = %d, want 0", id)
	}
}

func TestDRYSingleTokenBreakerMatchesPinnedUpstreamBehavior(t *testing.T) {
	base := Config{
		DryMultiplier:    2,
		DryBase:          1.1,
		DryAllowedLength: 2,
		DryPenaltyLastN:  6,
	}
	withoutBreaker, err := New(base)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{0, 0, 9, 10, 0}
	history := []int{0, 1, 3, 4, 0, 1}
	id, err := withoutBreaker.SampleWithHistory(logits, history)
	if err != nil {
		t.Fatal(err)
	}
	if id != 2 {
		t.Fatalf("DRY without breaker ID = %d, want 2", id)
	}

	base.DryBreakers = [][]int{{3}}
	withBreaker, err := New(base)
	if err != nil {
		t.Fatal(err)
	}
	id, err = withBreaker.SampleWithHistory(logits, history)
	if err != nil {
		t.Fatal(err)
	}
	if id != 3 {
		t.Fatalf("DRY single-token breaker ID = %d, want 3", id)
	}
}

func TestDRYBreakerConfigurationIsImmutable(t *testing.T) {
	breakers := [][]int{{3, 4}}
	sampler, err := New(Config{DryBreakers: breakers})
	if err != nil {
		t.Fatal(err)
	}
	breakers[0][0] = 9
	config := sampler.Config()
	if config.DryBreakers[0][0] != 3 {
		t.Fatalf("stored breaker changed to %v", config.DryBreakers)
	}
	config.DryBreakers[0][0] = 8
	if sampler.Config().DryBreakers[0][0] != 3 {
		t.Fatal("Config exposed mutable breaker storage")
	}
}

func TestDRYMatchesPinnedUpstreamCoreCase(t *testing.T) {
	sampler, err := New(Config{
		DryMultiplier:    1,
		DryBase:          1.1,
		DryAllowedLength: 2,
		DryPenaltyLastN:  5,
	})
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{0, 0, 0, 0}
	sampler.applyDry(logits, []int{0, 1, 2, 0, 1})
	want := []float32{0, 0, -1, 0}
	for index := range want {
		if logits[index] != want[index] {
			t.Fatalf("DRY logit[%d] = %v, want %v", index, logits[index], want[index])
		}
	}
}

func TestChoiceGrammarConstrainsGreedyAndSampling(t *testing.T) {
	grammar, err := NewChoiceGrammar(
		[][]int{{2, 4}, {3, 1}},
		[]int{5},
		6,
	)
	if err != nil {
		t.Fatal(err)
	}
	sampler, err := New(Config{Temperature: 1, TopP: 1, TopK: 1, Grammar: grammar})
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{100, 90, 1, 2, 80, 70}
	first, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if first != 3 {
		t.Fatalf("grammar first token = %d, want 3", first)
	}
	second, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if second != 1 {
		t.Fatalf("grammar second token = %d, want 1", second)
	}
	terminal, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if terminal != 5 {
		t.Fatalf("grammar terminal token = %d, want EOS 5", terminal)
	}
}

func TestChoiceGrammarResetAndConfigurationIsolation(t *testing.T) {
	grammar, err := NewChoiceGrammar([][]int{{1, 2}}, []int{3}, 4)
	if err != nil {
		t.Fatal(err)
	}
	sampler, err := New(Config{Grammar: grammar})
	if err != nil {
		t.Fatal(err)
	}
	grammar.Transitions[0][1] = -1
	if token, sampleErr := sampler.Sample([]float32{9, 1, 8, 7}); sampleErr != nil || token != 1 {
		t.Fatalf("isolated grammar sample = %d, %v", token, sampleErr)
	}
	sampler.Reset()
	if token, sampleErr := sampler.Sample([]float32{9, 1, 8, 7}); sampleErr != nil || token != 1 {
		t.Fatalf("reset grammar sample = %d, %v", token, sampleErr)
	}
	config := sampler.Config()
	config.Grammar.Transitions[0][1] = -1
	sampler.Reset()
	if token, sampleErr := sampler.Sample([]float32{9, 1, 8, 7}); sampleErr != nil || token != 1 {
		t.Fatalf("configuration exposed grammar state: %d, %v", token, sampleErr)
	}
}

func TestGBNFGrammarConstrainsTokensAndEOS(t *testing.T) {
	grammar, err := NewGBNFGrammar(
		`root ::= "a" ("b" | "c")`,
		"root",
		bytePieces("a", "b", "c", "", "x", "ab"),
		[]int{3},
	)
	if err != nil {
		t.Fatal(err)
	}
	sampler, err := New(Config{GBNF: grammar})
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{10, 30, 20, 100, 90, 5}
	first, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if first != 0 {
		t.Fatalf("GBNF first token = %d, want 0", first)
	}
	second, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if second != 1 {
		t.Fatalf("GBNF second token = %d, want 1", second)
	}
	terminal, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if terminal != 3 {
		t.Fatalf("GBNF terminal token = %d, want EOS 3", terminal)
	}

	sampler.Reset()
	logits[5] = 40
	combined, err := sampler.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if combined != 5 {
		t.Fatalf("GBNF combined token = %d, want 5", combined)
	}
}

func TestTokenGrammarAndGBNFAreMutuallyExclusive(t *testing.T) {
	tokenGrammar, err := NewChoiceGrammar([][]int{{0}}, []int{1}, 2)
	if err != nil {
		t.Fatal(err)
	}
	gbnf, err := NewGBNFGrammar(`root ::= "a"`, "root", bytePieces("a", ""), []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := New(Config{Grammar: tokenGrammar, GBNF: gbnf}); err == nil {
		t.Fatal("sampler accepted token grammar and GBNF together")
	}
}

func TestPostSamplingProbabilitiesReflectFilteredCandidates(t *testing.T) {
	sampler, err := New(Config{
		Temperature: 1,
		TopK:        2,
		TopP:        1,
		Seed:        7,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := sampler.SampleWithHistoryProbabilities(
		[]float32{3, 2, 1},
		nil,
		3,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Top) != 2 ||
		result.Top[0].ID != 0 ||
		result.Top[1].ID != 1 ||
		result.Top[0].Probability <= result.Top[1].Probability ||
		math.Abs(
			result.Top[0].Probability+result.Top[1].Probability-1,
		) > 1e-12 {
		t.Fatalf("result = %+v", result)
	}
	foundSelected := false
	for _, item := range result.Top {
		if item.ID == result.Token {
			foundSelected = true
			if item.Probability != result.SelectedProbability {
				t.Fatalf("selected probability = %g, top = %+v", result.SelectedProbability, item)
			}
		}
	}
	if !foundSelected {
		t.Fatalf("selected token %d missing from %+v", result.Token, result.Top)
	}

	greedy, err := New(Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	greedyResult, err := greedy.SampleWithHistoryProbabilities(
		[]float32{1, 4, 3},
		nil,
		3,
	)
	if err != nil {
		t.Fatal(err)
	}
	if greedyResult.Token != 1 ||
		greedyResult.SelectedProbability != 1 ||
		len(greedyResult.Top) != 1 ||
		greedyResult.Top[0].ID != 1 ||
		greedyResult.Top[0].Probability != 1 {
		t.Fatalf("greedy result = %+v", greedyResult)
	}
}
