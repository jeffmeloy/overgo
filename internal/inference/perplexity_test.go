package inference

import (
	"math"
	"testing"
)

func TestNegativeLogProbability(t *testing.T) {
	got, err := negativeLogProbability([]float32{1, 2, 3}, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Log(math.Exp(-2) + math.Exp(-1) + 1)
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("negative log probability = %.15g, want %.15g", got, want)
	}
}

func TestNegativeLogProbabilityIsStableAndRejectsInvalidData(t *testing.T) {
	got, err := negativeLogProbability([]float32{10000, 9999}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-math.Log1p(math.Exp(-1))) > 1e-12 {
		t.Fatalf("stable negative log probability = %.15g", got)
	}
	for _, test := range []struct {
		logits []float32
		target int
	}{
		{nil, 0},
		{[]float32{1}, -1},
		{[]float32{1}, 1},
		{[]float32{float32(math.NaN())}, 0},
		{[]float32{float32(math.Inf(-1))}, 0},
	} {
		if _, scoreErr := negativeLogProbability(test.logits, test.target); scoreErr == nil {
			t.Fatalf("invalid score input %#v/%d was accepted", test.logits, test.target)
		}
	}
}
