package testutil

import (
	"math"
	"testing"
)

// refusal stands in for a test and records that a comparison refused what it
// was given: its Fatalf returns, where a test's stops.
type refusal struct {
	testing.TB
	refused bool
}

func (r *refusal) Fatalf(string, ...any) { r.refused = true }

// TestEveryComparisonRefusesAValueThatIsNotANumber holds each comparison a
// golden test leans on to failing when it is handed a value that is not a
// number. Every ordered comparison with such a value is false, so a helper
// that only asks whether the difference exceeds its bound passes it, and a
// computed gradient that came back as not a number would pass its golden.
func TestEveryComparisonRefusesAValueThatIsNotANumber(t *testing.T) {
	t.Parallel()
	notANumber := []float32{float32(math.NaN())}
	for _, comparison := range []struct {
		name    string
		compare func(testing.TB)
	}{
		{"RequireSliceClose", func(tb testing.TB) { RequireSliceClose(tb, "x", notANumber, []float64{0}, 1) }},
		{"RequireWithin", func(tb testing.TB) { RequireWithin(tb, "x", notANumber, []float64{0}, 1) }},
		{"RequireElementsWithin, computed side", func(tb testing.TB) { RequireElementsWithin(tb, "x", notANumber, []float64{0}, 1) }},
		{"RequireElementsWithin, expected side", func(tb testing.TB) { RequireElementsWithin(tb, "x", []float32{0}, []float64{math.NaN()}, 1) }},
		{"RequireClose", func(tb testing.TB) { RequireClose(tb, "x", math.NaN(), 0, 1) }},
		{"RequireRange", func(tb testing.TB) { RequireRange(tb, "x", math.NaN(), 0, 1) }},
		{"RequireFiniteDecrease", func(tb testing.TB) { RequireFiniteDecrease(tb, "x", []float64{math.NaN()}, 1) }},
	} {
		recorded := &refusal{TB: t}
		comparison.compare(recorded)
		if !recorded.refused {
			t.Errorf("%s passed a value that is not a number", comparison.name)
		}
	}
}

// TestElementComparisonStillPassesWhatIsWithinItsBound holds the element
// comparison to passing a pair inside its bound and refusing one outside it,
// so that refusing a value that is not a number did not cost it the bound.
func TestElementComparisonStillPassesWhatIsWithinItsBound(t *testing.T) {
	t.Parallel()
	inside := &refusal{TB: t}
	RequireElementsWithin(inside, "x", []float32{0, 1}, []float64{0, 1}, 0)
	if inside.refused {
		t.Error("an exact match was refused")
	}
	outside := &refusal{TB: t}
	RequireElementsWithin(outside, "x", []float32{0, 1}, []float64{0, 0}, 0)
	if !outside.refused {
		t.Error("a second element one away from its golden passed a bound of zero")
	}
}
