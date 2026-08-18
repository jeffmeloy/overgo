package testutil

import (
	"math"
	"testing"
)

type Float interface {
	~float32 | ~float64
}

// MaxAbsDiff returns the maximum elementwise absolute difference.
func MaxAbsDiff[L, R Float](left []L, right []R) float64 {
	if len(left) != len(right) {
		return math.Inf(1)
	}
	worst := 0.0
	for index := range left {
		difference := math.Abs(float64(left[index]) - float64(right[index]))
		if math.IsNaN(difference) {
			return math.Inf(1)
		}
		worst = max(worst, difference)
	}
	return worst
}

// RequireSliceClose checks elementwise absolute tolerance.
func RequireSliceClose[L, R Float](t testing.TB, label string, got []L, want []R, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	if difference := MaxAbsDiff(got, want); difference > tolerance {
		t.Fatalf("%s max abs diff = %g, want <= %g", label, difference, tolerance)
	}
}

// RequireFiniteDecrease checks a fixed-length decreasing trajectory.
func RequireFiniteDecrease(t testing.TB, label string, values []float64, count int) {
	t.Helper()
	if len(values) != count {
		t.Fatalf("%s length = %d, want %d", label, len(values), count)
	}
	for index, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("%s[%d] is not finite: %g", label, index, value)
		}
		if index > 0 && value >= values[index-1] {
			t.Fatalf("%s did not decrease at %d: %v", label, index, values)
		}
	}
}

// RequireClose checks one absolute evidence bound.
func RequireClose(t testing.TB, label string, got, want, tolerance float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > tolerance {
		t.Fatalf("%s = %g, want %g +/- %g", label, got, want, tolerance)
	}
}

// RequireRange checks one inclusive evidence interval.
func RequireRange(t testing.TB, label string, got, minimum, maximum float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || got < minimum || got > maximum {
		t.Fatalf("%s = %g, want [%g, %g]", label, got, minimum, maximum)
	}
}
