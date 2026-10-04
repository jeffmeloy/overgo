package optimizer

import (
	"math"
	"testing"
)

// averageClosedForm is the average after the weights in sequence, from the
// weights it started at, written without the recurrence: the start's share is
// the product of every update's decay, and update i's weights keep a share of
// (1-d_i) times the decays of the updates after it.
func averageClosedForm(start float64, sequence []float64, decay float64) float64 {
	rate := func(update int) float64 {
		return min(decay, float64(1+update)/float64(10+update))
	}
	total := len(sequence)
	share := 1.0
	for update := 1; update <= total; update++ {
		share *= rate(update)
	}
	value := share * start
	for update := 1; update <= total; update++ {
		weight := 1 - rate(update)
		for later := update + 1; later <= total; later++ {
			weight *= rate(later)
		}
		value += weight * sequence[update-1]
	}
	return value
}

// TestWeightAverageEqualsItsClosedForm checks the recurrence against the sum it
// stands for, under a decay the warm-up holds below for the first updates and
// under one it caps from the first, and that a constant weight is its own average.
func TestWeightAverageEqualsItsClosedForm(t *testing.T) {
	sequence := make([]float64, 40)
	for index := range sequence {
		sequence[index] = math.Sin(float64(index)) + 0.1*float64(index)
	}
	for _, decay := range []float64{0.99, 0.5, 0.1} {
		average, err := NewAverage([]float32{2, -1}, decay)
		if err != nil {
			t.Fatal(err)
		}
		for _, weight := range sequence {
			if err := average.Update([]float32{float32(weight), float32(-weight)}); err != nil {
				t.Fatal(err)
			}
		}
		for index, start := range []float64{2, -1} {
			signed := make([]float64, len(sequence))
			for at, weight := range sequence {
				signed[at] = weight * []float64{1, -1}[index]
			}
			if want, got := averageClosedForm(start, signed, decay), float64(average.Values()[index]); math.Abs(want-got) > 1e-5 {
				t.Errorf("decay %v, entry %d: average %.7f, closed form %.7f", decay, index, got, want)
			}
		}
		if average.Updates() != len(sequence) {
			t.Errorf("decay %v: %d updates counted, want %d", decay, average.Updates(), len(sequence))
		}
	}
	constant, err := NewAverage([]float32{3}, 0.99)
	if err != nil {
		t.Fatal(err)
	}
	for range 25 {
		if err := constant.Update([]float32{3}); err != nil {
			t.Fatal(err)
		}
	}
	if got := constant.Values()[0]; math.Abs(float64(got)-3) > 1e-6 {
		t.Errorf("an average of constant weights is %v, want 3", got)
	}
}

// TestWeightAverageWarmsUp checks the first update moves the average most of the
// way to the weights, as the warm-up promises, where the bare decay of 0.99 would
// leave it nearly where it began.
func TestWeightAverageWarmsUp(t *testing.T) {
	average, err := NewAverage([]float32{0}, 0.99)
	if err != nil {
		t.Fatal(err)
	}
	if err := average.Update([]float32{1}); err != nil {
		t.Fatal(err)
	}
	if got := average.Values()[0]; math.Abs(float64(got)-9.0/11) > 1e-6 {
		t.Fatalf("after one update the average is %v, want 9/11 (decay 2/11)", got)
	}
}

// TestWeightAverageRefusesWhatItCannotAverage checks decays outside (0, 1) and
// weights of another length are errors, and the start is copied, not aliased.
func TestWeightAverageRefusesWhatItCannotAverage(t *testing.T) {
	for _, decay := range []float64{0, 1, -0.5, 1.5, math.NaN()} {
		if _, err := NewAverage([]float32{1}, decay); err == nil {
			t.Errorf("a decay of %v was accepted", decay)
		}
	}
	start := []float32{1, 2}
	average, err := NewAverage(start, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	if err := average.Update([]float32{1}); err == nil {
		t.Error("weights of another length were accepted")
	}
	start[0] = 99
	if average.Values()[0] != 1 {
		t.Error("the average aliases the weights it started from")
	}
}
