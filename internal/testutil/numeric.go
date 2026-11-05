package testutil

import (
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testcheck"
)

// Map is each of values converted, in their order.
func Map[In, Out any](values []In, convert func(In) Out) []Out {
	converted := make([]Out, len(values))
	for index, value := range values {
		converted[index] = convert(value)
	}
	return converted
}

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

// RequireWithin fails the test unless got matches want elementwise
// within tol; the max-abs-diff observation is logged either way so
// golden gates leave a numeric trace.
func RequireWithin[L, R Float](t testing.TB, name string, got []L, want []R, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != golden %d", name, len(got), len(want))
	}
	diff := MaxAbsDiff(got, want)
	t.Logf("%s: max abs diff %.6e (gate %.0e)", name, diff, tol)
	if diff > tol {
		t.Fatalf("%s diverges: %g > %g", name, diff, tol)
	}
}

// RequireElementsWithin fails at the first element that differs by more than
// tol or is not a number, naming the index for golden triage.
func RequireElementsWithin[L, R Float](t testing.TB, name string, got []L, want []R, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if d := MaxAbsDiff(got[i:i+1], want[i:i+1]); d > tol {
			t.Fatalf("%s[%d]: |%g - %g| = %g > %g", name, i, got[i], want[i], d, tol)
		}
	}
}

// ModelArtifactDir resolves one named model directory under the
// repository data roots for artifact-gated golden tests: the first model
// root that holds it, and the configured models root, where a test reports it
// absent, when none does.
func ModelArtifactDir(t testing.TB, name string) string {
	t.Helper()
	roots, err := dataroot.Resolve(RepoRoot(t))
	testcheck.NoError(t, err)
	if resolved := roots.ResolveModelPath(name); filepath.IsAbs(resolved) {
		return resolved
	}
	return filepath.Join(roots.Models, name)
}

// NormalVector draws count standard normal values from source, each times scale.
func NormalVector(source interface{ NormFloat64() float64 }, count int, scale float64) []float32 {
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(source.NormFloat64() * scale)
	}
	return values
}

// NormalVectorAbout draws count values of base plus scale times a standard normal draw from source.
func NormalVectorAbout(source interface{ NormFloat64() float64 }, count int, base, scale float64) []float32 {
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(base + source.NormFloat64()*scale)
	}
	return values
}

// Float32s narrows each value to float32.
func Float32s(values []float64) []float32 {
	narrowed := make([]float32, len(values))
	for index, value := range values {
		narrowed[index] = float32(value)
	}
	return narrowed
}

// The xorshift32 generator's three shifts (Marsaglia 2003), and how XorShift folds a state into a draw: its
// remainder among xorShiftBuckets, over xorShiftPerUnit, less one, times xorShiftScale.
const (
	xorShiftFirst, xorShiftSecond, xorShiftThird = 13, 17, 5
	xorShiftBuckets                              = 2000
	xorShiftPerUnit                              = 1000
	xorShiftScale                                = 0.3
)

// XorShift is the xorshift32 stream from seed, each draw a value in [-0.3, 0.3) at a step of 0.0003.
func XorShift(seed uint32) func() float32 {
	return func() float32 {
		seed ^= seed << xorShiftFirst
		seed ^= seed >> xorShiftSecond
		seed ^= seed << xorShiftThird
		return (float32(seed%xorShiftBuckets)/xorShiftPerUnit - 1) * xorShiftScale
	}
}
