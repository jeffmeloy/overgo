package tensor

import (
	"math"
	"testing"
)

// TestQSARopeTableIsEachPositionsRotation holds the QSA rotation table, its positions computed in parallel spans, to
// rope.cu's build_rope_table value by value and bit for bit: cos and sin of position p times 1e7^(-2i / 64) in float64,
// rounded to float32, at every position of a table longer than any processor's span.
func TestQSARopeTableIsEachPositionsRotation(t *testing.T) {
	t.Parallel()
	const positions = 4099
	half := QSARotated / PairedExtent
	cosines, sines := QSARopeTable(positions)
	if len(cosines) != positions*half || len(sines) != positions*half {
		t.Fatalf("a table of %d positions holds %d cosines and %d sines, want %d each", positions, len(cosines), len(sines), positions*half)
	}
	for position := range positions {
		for pair := range half {
			angle := float64(position) * math.Pow(QSARopeBase, -PairedExtent*float64(pair)/float64(QSARotated))
			at := position*half + pair
			if math.Float32bits(cosines[at]) != math.Float32bits(float32(math.Cos(angle))) ||
				math.Float32bits(sines[at]) != math.Float32bits(float32(math.Sin(angle))) {
				t.Fatalf("position %d pair %d holds cos %g and sin %g, want %g and %g", position, pair, cosines[at], sines[at], math.Cos(angle), math.Sin(angle))
			}
		}
	}
}
