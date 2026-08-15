package sampling

import (
	"encoding/binary"
	"math"
	"slices"
	"testing"
)

func TestSamplerStateResumesExactly(t *testing.T) {
	config := Config{
		Temperature: 0.8,
		TopK:        4,
		TopP:        0.9,
		Mirostat:    2,
		MirostatTau: 3,
		MirostatEta: 0.2,
		Seed:        42,
	}
	original, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{1, 2, 3, 4, 5}
	for range 10 {
		if _, err := original.Sample(logits); err != nil {
			t.Fatal(err)
		}
	}
	state, err := original.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	expected := make([]int, 20)
	for index := range expected {
		expected[index], err = original.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
	}
	expectedMu := original.mu

	resumed, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.LoadState(state); err != nil {
		t.Fatal(err)
	}
	actual := make([]int, len(expected))
	for index := range actual {
		actual[index], err = resumed.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
	}
	if !slices.Equal(actual, expected) {
		t.Fatalf("resumed IDs = %v, want %v", actual, expected)
	}
	if resumed.mu != expectedMu {
		t.Fatalf("resumed mu = %v, want %v", resumed.mu, expectedMu)
	}
}

func TestSamplerStateRejectsMalformedOrMismatchedData(t *testing.T) {
	sampler, err := New(Config{Temperature: 1, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	state, err := sampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	for _, malformed := range [][]byte{
		nil,
		state[:31],
		append(append([]byte(nil), state...), 0),
		append([]byte("BADSTATE"), state[8:]...),
	} {
		if err := sampler.LoadState(malformed); err == nil {
			t.Fatalf("malformed state of length %d was accepted", len(malformed))
		}
	}
	different, err := New(Config{Temperature: 0.5, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if err := different.LoadState(state); err == nil {
		t.Fatal("state with a different configuration was accepted")
	}
	invalidMu := append([]byte(nil), state...)
	binary.LittleEndian.PutUint64(invalidMu[24:], math.Float64bits(math.NaN()))
	if err := sampler.LoadState(invalidMu); err == nil {
		t.Fatal("state with NaN Mirostat mu was accepted")
	}
}

func TestSamplerOrderParticipatesInStateSignature(t *testing.T) {
	implicit, err := New(Config{Temperature: 1, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	explicitDefault, err := New(Config{
		Temperature: 1,
		Seed:        7,
		Samplers:    DefaultSamplerOrder(),
	})
	if err != nil {
		t.Fatal(err)
	}
	custom, err := New(Config{
		Temperature: 1,
		Seed:        7,
		Samplers:    []SamplerStage{SamplerTemperature},
	})
	if err != nil {
		t.Fatal(err)
	}
	if configSignature(implicit.config) != configSignature(explicitDefault.config) {
		t.Fatal("explicit default order changed the legacy configuration signature")
	}
	state, err := implicit.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if err := explicitDefault.LoadState(state); err != nil {
		t.Fatalf("explicit default rejected default state: %v", err)
	}
	if err := custom.LoadState(state); err == nil {
		t.Fatal("custom order accepted default-order state")
	}
}

func TestXTCStateResumesRNGExactly(t *testing.T) {
	config := Config{
		Temperature:    1,
		TopP:           1,
		XTCProbability: 0.5,
		XTCThreshold:   0.15,
		Seed:           42,
	}
	original, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{4, 3, 2, 1}
	for range 7 {
		if _, err := original.Sample(logits); err != nil {
			t.Fatal(err)
		}
	}
	state, err := original.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	want := make([]int, 20)
	for index := range want {
		want[index], err = original.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
	}
	resumed, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.LoadState(state); err != nil {
		t.Fatal(err)
	}
	got := make([]int, len(want))
	for index := range got {
		got[index], err = resumed.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
	}
	if !slices.Equal(got, want) {
		t.Fatalf("resumed XTC IDs = %v, want %v", got, want)
	}
}

func TestDynamicTemperatureParticipatesInStateSignature(t *testing.T) {
	fixed, err := New(Config{Temperature: 0.8, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	dynamic, err := New(Config{
		Temperature:      0.8,
		DynatempRange:    0.2,
		DynatempExponent: 1.5,
		Seed:             7,
	})
	if err != nil {
		t.Fatal(err)
	}
	state, err := fixed.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if err := dynamic.LoadState(state); err == nil {
		t.Fatal("dynamic-temperature sampler accepted fixed-temperature state")
	}
}

func TestAdaptivePStateResumesEMAAndRNGExactly(t *testing.T) {
	config := Config{
		AdaptiveTarget: 0.2,
		AdaptiveDecay:  0.8,
		Seed:           42,
		Samplers:       []SamplerStage{SamplerMinP, SamplerAdaptiveP},
		MinP:           0.01,
	}
	original, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{
		float32(math.Log(0.6)),
		float32(math.Log(0.25)),
		float32(math.Log(0.1)),
		float32(math.Log(0.05)),
	}
	for range 9 {
		if _, err := original.Sample(logits); err != nil {
			t.Fatal(err)
		}
	}
	state, err := original.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	want := make([]int, 20)
	for index := range want {
		want[index], err = original.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
	}
	wantSum, wantWeight := original.adaptiveSum, original.adaptiveWeight

	resumed, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.LoadState(state); err != nil {
		t.Fatal(err)
	}
	got := make([]int, len(want))
	for index := range got {
		got[index], err = resumed.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
	}
	if !slices.Equal(got, want) {
		t.Fatalf("resumed adaptive-p IDs = %v, want %v", got, want)
	}
	if resumed.adaptiveSum != wantSum || resumed.adaptiveWeight != wantWeight {
		t.Fatalf(
			"resumed adaptive EMA = %v/%v, want %v/%v",
			resumed.adaptiveSum,
			resumed.adaptiveWeight,
			wantSum,
			wantWeight,
		)
	}
	resumed.Reset()
	denominator := 1 - float64(config.AdaptiveDecay)
	if resumed.adaptiveSum != float64(config.AdaptiveTarget)/denominator ||
		resumed.adaptiveWeight != 1/denominator {
		t.Fatalf(
			"reset adaptive EMA = %v/%v",
			resumed.adaptiveSum,
			resumed.adaptiveWeight,
		)
	}
}

func TestLogitBiasParticipatesInStateSignature(t *testing.T) {
	plain, err := New(Config{Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	biased, err := New(Config{
		Seed:        7,
		LogitBiases: []LogitBias{{Token: 1, Bias: -2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	state, err := plain.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if err := biased.LoadState(state); err == nil {
		t.Fatal("biased sampler accepted unbiassed state")
	}
}

func TestInfillVocabularyParticipatesInStateSignature(t *testing.T) {
	config := Config{
		Seed:     7,
		Samplers: []SamplerStage{SamplerInfill},
		Infill: &InfillVocabulary{
			Pieces: []string{"a", "</s>"},
			EOG:    []bool{false, true},
			EOT:    1,
			EOS:    1,
		},
	}
	source, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	state, err := source.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	config.Infill.Pieces[0] = "b"
	different, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := different.LoadState(state); err == nil {
		t.Fatal("infill sampler accepted state for a different vocabulary")
	}
}

func TestSamplerGrammarStateResumesExactly(t *testing.T) {
	grammar, err := NewChoiceGrammar(
		[][]int{{1, 2, 3}, {1, 4, 3}},
		[]int{5},
		6,
	)
	if err != nil {
		t.Fatal(err)
	}
	config := Config{Grammar: grammar}
	original, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{100, 10, 9, 8, 11, 7}
	if token, sampleErr := original.Sample(logits); sampleErr != nil || token != 1 {
		t.Fatalf("first grammar token = %d, %v", token, sampleErr)
	}
	state, err := original.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	want, err := original.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.LoadState(state); err != nil {
		t.Fatal(err)
	}
	got, err := resumed.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || got != 4 {
		t.Fatalf("resumed grammar token = %d, want %d", got, want)
	}
}

func TestSamplerGBNFStateResumesExactly(t *testing.T) {
	grammar, err := NewGBNFGrammar(
		`root ::= "a" ("b" | "c")`,
		"root",
		bytePieces("a", "b", "c", "", "x"),
		[]int{3},
	)
	if err != nil {
		t.Fatal(err)
	}
	config := Config{GBNF: grammar}
	original, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{10, 8, 9, 100, 90}
	if token, sampleErr := original.Sample(logits); sampleErr != nil || token != 0 {
		t.Fatalf("first GBNF token = %d, %v", token, sampleErr)
	}
	state, err := original.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	want, err := original.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.LoadState(state); err != nil {
		t.Fatal(err)
	}
	got, err := resumed.Sample(logits)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || got != 2 {
		t.Fatalf("resumed GBNF token = %d, want %d", got, want)
	}

	tampered := append([]byte(nil), state...)
	binary.LittleEndian.PutUint32(tampered[samplerStateHeaderSize:], 4)
	if err := resumed.LoadState(tampered); err == nil {
		t.Fatal("sampler accepted GBNF history rejected by its grammar")
	}
}

func TestSamplerLazyGBNFStateRestoresBufferedTriggerHistory(t *testing.T) {
	grammar, err := NewGBNFGrammarWithOptions(
		`root ::= "JSON:" [0-9]+`,
		"root",
		bytePieces("prefixJS", "ON:42", "", "other"),
		[]int{2},
		nil,
		GBNFLazyOptions{
			Enabled:  true,
			Patterns: []string{`prefix(JSON:)`},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	config := Config{GBNF: grammar}
	source, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	firstLogits := []float32{10, 1, 0, 9}
	if token, sampleErr := source.Sample(firstLogits); sampleErr != nil || token != 0 {
		t.Fatalf("lazy GBNF prefix token = %d, %v", token, sampleErr)
	}
	state, err := source.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.LoadState(state); err != nil {
		t.Fatal(err)
	}
	triggerLogits := []float32{0, 10, 1, 9}
	want, err := source.Sample(triggerLogits)
	if err != nil {
		t.Fatal(err)
	}
	got, err := resumed.Sample(triggerLogits)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || got != 1 ||
		source.gbnfState.awaitingTrigger ||
		resumed.gbnfState.awaitingTrigger {
		t.Fatalf("restored lazy GBNF trigger = %d, want %d", got, want)
	}
	resumed.Reset()
	if !resumed.gbnfState.awaitingTrigger ||
		len(resumed.gbnfState.triggerBuffer) != 0 {
		t.Fatal("reset did not restore lazy GBNF trigger state")
	}
}
