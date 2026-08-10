package server

import (
	"math"
	"testing"
)

// The entropy attached to the raw-logits probability output is the full-vocab
// Shannon entropy (nats) of the model's next-token distribution — a
// distribution-free measurement, exercised here against closed-form values.
func TestNativeTokenProbabilityEntropy(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})

	cases := []struct {
		name   string
		logits []float32
		want   float64
	}{
		// Uniform over 2 tokens → ln 2.
		{"uniform2", []float32{0, 0}, math.Ln2},
		// Uniform over 4 tokens → ln 4 = 2 ln 2.
		{"uniform4", []float32{1, 1, 1, 1}, 2 * math.Ln2},
		// Near-deterministic → entropy ≈ 0.
		{"peaked", []float32{100, 0}, 0},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			result, err := handler.nativeTokenProbability(0, testCase.logits, 2)
			if err != nil {
				t.Fatalf("nativeTokenProbability: %v", err)
			}
			if result.Entropy == nil {
				t.Fatal("entropy not reported")
			}
			if math.Abs(*result.Entropy-testCase.want) > 1e-6 {
				t.Fatalf("entropy = %v, want %v", *result.Entropy, testCase.want)
			}
		})
	}
}

// Entropy is bounded above by ln(vocabulary size) and is maximized by the
// uniform distribution — a property that must hold without any distributional
// assumption. A skewed distribution must have strictly lower entropy than the
// uniform one over the same support.
func TestNativeTokenProbabilityEntropyBoundedByUniform(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	uniform, err := handler.nativeTokenProbability(0, []float32{0, 0, 0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	skewed, err := handler.nativeTokenProbability(0, []float32{3, 0, 0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	maxEntropy := math.Log(4)
	if *uniform.Entropy > maxEntropy+1e-9 {
		t.Fatalf("uniform entropy %v exceeds ln(4) %v", *uniform.Entropy, maxEntropy)
	}
	if !(*skewed.Entropy < *uniform.Entropy) {
		t.Fatalf("skewed entropy %v not < uniform %v", *skewed.Entropy, *uniform.Entropy)
	}
}
