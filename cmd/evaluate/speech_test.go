package main

import (
	"math"
	"testing"
)

// tone is n samples of a sine of the given frequency at a sample rate.
func tone(frequency float64, rate, n int) []float32 {
	samples := make([]float32, n)
	for index := range samples {
		samples[index] = float32(math.Sin(2 * math.Pi * frequency * float64(index) / float64(rate)))
	}
	return samples
}

// TestResamplingKeepsATonesFrequencyAndLevelAndDropsWhatTheLowerRateCannotHold
// holds the resampler to what a recognizer's input must be: a 1 kHz tone
// converted from 24 kHz to 16 kHz is the same tone at the lower rate to within a
// couple of percent, the output has the length the ratio gives, and a tone
// above the lower rate's Nyquist frequency is attenuated, not folded back.
func TestResamplingKeepsATonesFrequencyAndLevelAndDropsWhatTheLowerRateCannotHold(t *testing.T) {
	t.Parallel()
	const from, to, samples = 24000, 16000, 24000
	converted, err := resampleTo(t.Context(), tone(1000, from, samples), from, to)
	if err != nil {
		t.Fatal(err)
	}
	if want := samples * to / from; len(converted) != want {
		t.Fatalf("converted %d samples to %d, want %d", samples, len(converted), want)
	}
	want := tone(1000, to, len(converted))
	// Away from the edges the filter's start-up and tail disturb.
	var worst float64
	for index := len(converted) / 4; index < 3*len(converted)/4; index++ {
		worst = math.Max(worst, math.Abs(float64(converted[index]-want[index])))
	}
	if worst > 0.02 {
		t.Fatalf("a 1 kHz tone differs from the same tone at %d Hz by %.4f at most, want under 0.02", to, worst)
	}
	above, err := resampleTo(t.Context(), tone(11000, from, samples), from, to)
	if err != nil {
		t.Fatal(err)
	}
	var peak float64
	for index := len(above) / 4; index < 3*len(above)/4; index++ {
		peak = math.Max(peak, math.Abs(float64(above[index])))
	}
	if peak > 0.05 {
		t.Fatalf("an 11 kHz tone, past the %d Hz Nyquist frequency, leaves a peak of %.4f, want it filtered out", to/2, peak)
	}
}

// TestResamplingToTheSameRateIsTheSamplesAndAnInvalidRateIsRefused holds the
// edges: nothing to convert returns the samples, and a rate that is not positive
// is refused.
func TestResamplingToTheSameRateIsTheSamplesAndAnInvalidRateIsRefused(t *testing.T) {
	t.Parallel()
	samples := tone(440, 16000, 100)
	same, err := resampleTo(t.Context(), samples, 16000, 16000)
	if err != nil || len(same) != len(samples) || &same[0] != &samples[0] {
		t.Fatalf("same rate returned %d samples (%v), want the samples themselves", len(same), err)
	}
	for _, rates := range [][2]int{{0, 16000}, {24000, 0}, {-1, 16000}} {
		if _, err := resampleTo(t.Context(), samples, rates[0], rates[1]); err == nil {
			t.Errorf("a resample from %d Hz to %d Hz was accepted", rates[0], rates[1])
		}
	}
}
