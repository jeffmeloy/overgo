package inference

import (
	"math"
	"testing"

	"overgo/internal/model"
)

func TestApplyLogitSoftcap(t *testing.T) {
	logits := []float32{-100, -10, 0, 10, 100}
	got := applyLogitSoftcap(logits, 10)
	for index, want := range []float32{
		-10,
		-7.6159415,
		0,
		7.6159415,
		10,
	} {
		if difference := math.Abs(float64(got[index] - want)); difference > 0.001 {
			t.Fatalf(
				"softcap[%d] = %v, want %v (difference %v)",
				index,
				got[index],
				want,
				difference,
			)
		}
	}
	plain := []float32{-2, 3}
	if result := applyLogitSoftcap(plain, 0); &result[0] != &plain[0] ||
		result[0] != -2 ||
		result[1] != 3 {
		t.Fatalf("disabled softcap changed logits: %v", result)
	}
}

func TestChameleonImageLogitsAreSuppressed(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "chameleon", VocabularySize: 8200}}}}
	runner = attachFixtureProgram(runner)
	logits := make([]float32, 2*8200)
	for index := range logits {
		logits[index] = float32(index)
	}
	runner.finalizeLogits(logits)
	for _, base := range []int{0, 8200} {
		if logits[base+3] == -math.MaxFloat32 || logits[base+8196] == -math.MaxFloat32 {
			t.Fatal("text logit was suppressed")
		}
		if logits[base+4] != -math.MaxFloat32 || logits[base+8195] != -math.MaxFloat32 {
			t.Fatal("image-token logit was not suppressed")
		}
	}
}

func TestScaleLogits(t *testing.T) {
	logits := []float32{-2, 0.5, 4}
	scaleLogits(logits, 0.25)
	want := []float32{-0.5, 0.125, 1}
	for index := range want {
		if logits[index] != want[index] {
			t.Fatalf("scaled logit %d = %v, want %v", index, logits[index], want[index])
		}
	}
}

func TestAddOutputBiasBroadcastsAcrossTokens(t *testing.T) {
	logits := []float32{1, 2, 3, 4, 5, 6}
	if err := addOutputBias(logits, []float32{0.5, -1, 2}); err != nil {
		t.Fatal(err)
	}
	want := []float32{1.5, 1, 5, 4.5, 4, 8}
	for index := range want {
		if logits[index] != want[index] {
			t.Fatalf("biased logits[%d] = %v, want %v", index, logits[index], want[index])
		}
	}
	if err := addOutputBias([]float32{1, 2}, []float32{1, 2, 3}); err == nil {
		t.Fatal("incompatible output bias was accepted")
	}
}

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
