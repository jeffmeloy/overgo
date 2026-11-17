package latentimage

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

// TestANegativePromptsWeightsReachItsRows holds a weighted negative prompt in Krea's guided step, on the host
// reference, as NAG of Z+ against itself is Z+: a weighted prompt guided by itself, the guide carrying the prompt's
// weights, steps as the weighted prompt unguided does; the same negative prompt with its weights left off is another
// negative and moves the step; and a guide whose weights are not one per row is refused.
func TestANegativePromptsWeightsReachItsRows(t *testing.T) {
	t.Parallel()
	mask, weights := []bool{true, true, true}, []float32{1, 3, 1}
	guided := newGuidedStep(t, mask, weights)
	unguided := guided.unguided(t)
	weighed := guided.compile(t, NegativeGuide{Mask: mask, Weights: weights, Attention: guided.attention})
	if gap := velocityGap(t, guided.velocity(t, weighed, guided.prompt), unguided); gap > 1e-6 {
		t.Fatalf("the weighted prompt as its own weighted negative moved the velocity by %.3e; its weights did not reach the negative prompt's rows as they reach the prompt's", gap)
	}
	unweighed := guided.compile(t, NegativeGuide{Mask: mask, Attention: guided.attention})
	if gap := velocityGap(t, guided.velocity(t, unweighed, guided.prompt), unguided); gap < 1e-4 {
		t.Fatalf("the prompt as its own negative with its weights left off moved the velocity by %.3e; unweighted, it is another negative", gap)
	}
	if _, err := CompileDenoiserProgram(guided.spec, 1e-5, mask, weights, guidedHeight, guidedWidth, dtype.F32, false, NegativeGuide{Mask: mask, Weights: weights[:2], Attention: guided.attention}); err == nil {
		t.Fatal("a negative guide with two weights for three rows compiled")
	}
}
