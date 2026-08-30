package composition

import (
	"strings"
	"testing"
)

// seamSamples generates deterministic pseudo-activations wide enough to
// determine a linear map, using a fixed linear congruential stream so the
// fixture is reproducible without process randomness.
func seamSamples(samples, width int, seed uint64) [][]float64 {
	state := seed
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(int64(state>>33))/float64(1<<30) - 1
	}
	rows := make([][]float64, samples)
	for row := range rows {
		rows[row] = make([]float64, width)
		for column := range rows[row] {
			rows[row][column] = next()
		}
	}
	return rows
}

func applyLinearMap(source [][]float64, adapter [][]float64, noise func(row, column int) float64) [][]float64 {
	width := len(adapter[0])
	target := make([][]float64, len(source))
	for row := range source {
		target[row] = make([]float64, width)
		for column := 0; column < width; column++ {
			total := 0.0
			for k := range source[row] {
				total += source[row][k] * adapter[k][column]
			}
			target[row][column] = total + noise(row, column)
		}
	}
	return target
}

// TestSeamAlignmentResidual pins the proxy contract: a seam whose target
// representation is an exact linear image of the source has residual
// zero — a linear interface adapter composes it — the residual grows
// monotonically with the unexplainable part, unrelated representations
// approach one, and underdetermined, mismatched, or non-finite
// activations refuse instead of inventing a map.
func TestSeamAlignmentResidual(t *testing.T) {
	source := seamSamples(48, 6, 11)
	adapter := [][]float64{
		{1, 0.5, 0, 0}, {0, 1, -0.25, 0}, {0.5, 0, 1, 0.75},
		{0, -0.5, 0, 1}, {0.25, 0, 0.5, 0}, {0, 0.25, 0, -0.5},
	}
	exact, err := SeamAlignmentResidual(source, applyLinearMap(source, adapter, func(int, int) float64 { return 0 }))
	if err != nil {
		t.Fatal(err)
	}
	if exact > 1e-9 {
		t.Fatalf("linearly composable seam residual = %v", exact)
	}
	noiseAt := func(scale float64) func(int, int) float64 {
		stream := seamSamples(48, 4, 977)
		return func(row, column int) float64 { return scale * stream[row][column] }
	}
	small, err := SeamAlignmentResidual(source, applyLinearMap(source, adapter, noiseAt(0.1)))
	if err != nil {
		t.Fatal(err)
	}
	large, err := SeamAlignmentResidual(source, applyLinearMap(source, adapter, noiseAt(1.5)))
	if err != nil {
		t.Fatal(err)
	}
	if !(exact < small && small < large && large < 1) {
		t.Fatalf("residual is not monotone in the unexplainable part: %v %v %v", exact, small, large)
	}
	unrelated, err := SeamAlignmentResidual(source, seamSamples(48, 4, 5077))
	if err != nil {
		t.Fatal(err)
	}
	if unrelated < 0.5 {
		t.Fatalf("unrelated representations look composable: %v", unrelated)
	}

	if _, err := SeamAlignmentResidual(source[:5], applyLinearMap(source, adapter, func(int, int) float64 { return 0 })); err == nil {
		t.Fatal("mismatched sample counts fitted")
	}
	if _, err := SeamAlignmentResidual(source[:6], applyLinearMap(source[:6], adapter, func(int, int) float64 { return 0 })); err == nil ||
		!strings.Contains(err.Error(), "cannot determine") {
		t.Fatalf("underdetermined fit accepted: %v", err)
	}
	degenerate := seamSamples(48, 6, 11)
	for row := range degenerate {
		copy(degenerate[row], degenerate[0])
	}
	if _, err := SeamAlignmentResidual(degenerate, applyLinearMap(degenerate, adapter, func(int, int) float64 { return 0 })); err == nil ||
		!strings.Contains(err.Error(), "do not span") {
		t.Fatalf("rank-deficient activations fitted: %v", err)
	}
}

// TestAlignmentResidualBiasAudit pins the audit contract: the proxy is
// admitted only when its ordering against measured bridge parity holds
// with a violation fraction under the bound derived from the comparison
// count alone; a proxy ordered against parity refuses, too few
// comparisons make the bound unreachable and refuse admission, and ties
// never count as evidence either way.
func TestAlignmentResidualBiasAudit(t *testing.T) {
	residuals := make([]float64, 30)
	parities := make([]float64, 30)
	for index := range residuals {
		residuals[index] = float64(index) / 30
		parities[index] = 1 - residuals[index]
	}
	audit, err := AlignmentResidualBiasAudit(residuals, parities)
	if err != nil {
		t.Fatal(err)
	}
	if !audit.Admitted || audit.Violations != 0 || audit.Comparisons == 0 || audit.Bound <= 0 {
		t.Fatalf("perfectly predictive proxy refused: %+v", audit)
	}

	inverted, err := AlignmentResidualBiasAudit(residuals, residuals[:len(residuals)])
	if err != nil {
		t.Fatal(err)
	}
	if inverted.Admitted || inverted.Violations != inverted.Comparisons {
		t.Fatalf("anti-predictive proxy admitted: %+v", inverted)
	}

	tiny, err := AlignmentResidualBiasAudit([]float64{0.1, 0.2, 0.3}, []float64{0.9, 0.8, 0.7})
	if err != nil {
		t.Fatal(err)
	}
	if tiny.Admitted || tiny.Bound > 0 {
		t.Fatalf("three seams admitted a proxy: %+v", tiny)
	}

	tied, err := AlignmentResidualBiasAudit([]float64{0.1, 0.1, 0.3}, []float64{0.9, 0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if tied.Ties != 2 || tied.Comparisons != 1 {
		t.Fatalf("tie accounting = %+v", tied)
	}

	if _, err := AlignmentResidualBiasAudit([]float64{0.1}, []float64{0.9}); err == nil {
		t.Fatal("single measurement audited")
	}
}
