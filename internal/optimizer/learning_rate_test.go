package optimizer

import (
	"math"
	"testing"
)

func TestDeriveBaseLRInverseSqrtScaling(t *testing.T) {
	if got := DeriveBaseLR(0); got != 0 {
		t.Fatalf("DeriveBaseLR(0) = %g, want 0", got)
	}
	if got := DeriveBaseLR(-7); got != 0 {
		t.Fatalf("DeriveBaseLR(-7) = %g, want 0", got)
	}
	for _, n := range []int{1, 100, 5712, 500_000_000} {
		want := math.Pow(float64(n), BaseLRParamExponent)
		if got := DeriveBaseLR(n); got != want {
			t.Fatalf("DeriveBaseLR(%d) = %g, want %g", n, got, want)
		}
	}
	// 1/sqrt(n): the derived rate shrinks as the model grows, which is why one
	// derived rate holds from a tiny fixture to a 500M model.
	if !(DeriveBaseLR(500_000_000) < DeriveBaseLR(5712)) {
		t.Fatal("derived base LR must shrink as parameter count grows")
	}
}

func TestDeriveMomentumUsesEffectiveSampleRule(t *testing.T) {
	momentum := DeriveMomentum()
	wantNumerator := float64(CLTMinSamples - 1)
	if got := momentum * float64(CLTMinSamples+1); got != wantNumerator {
		t.Fatalf("momentum numerator = %g, want %g", got, wantNumerator)
	}
}
