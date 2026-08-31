package composition

import (
	"errors"
	"fmt"
	"math"
)

const (
	// alignmentAuditDenominator bounds the chance that a proxy with no
	// real relation to bridge parity passes the audit by luck at one in a
	// thousand; the admissible violation fraction derives from it and the
	// comparison count, never from a hand-picked threshold.
	alignmentAuditDenominator = 1000
)

// SeamAlignmentResidual measures how much of the target seam
// representation a linear adapter cannot reproduce from the source seam:
// the relative least-squares residual of the best linear map fitted over
// paired bounded seam activations — never whole checkpoints. Zero means a
// linear interface adapter composes the seam exactly; one means the seam
// representations share nothing linear. The fit refuses when the samples
// cannot determine the map.
func SeamAlignmentResidual(source, target [][]float64) (float64, error) {
	_, residual, err := AlignSeamAdapter(source, target)
	return residual, err
}

// AlignSeamAdapter fits the best linear interface adapter over paired
// bounded seam activations and returns it with its relative residual: the
// align-init a candidate realization starts training from. Refusals are
// exactly those of the residual computation.
func AlignSeamAdapter(source, target [][]float64) ([][]float64, float64, error) {
	if len(source) == 0 || len(source) != len(target) {
		return nil, 0, errors.New("composition: alignment residual requires paired seam activations")
	}
	samples, sourceWidth, targetWidth := len(source), len(source[0]), len(target[0])
	if sourceWidth == 0 || targetWidth == 0 {
		return nil, 0, errors.New("composition: alignment residual requires nonempty activation vectors")
	}
	if samples <= sourceWidth {
		return nil, 0, fmt.Errorf(
			"composition: %d paired samples cannot determine a %d-wide linear map", samples, sourceWidth,
		)
	}
	for row := range samples {
		if len(source[row]) != sourceWidth || len(target[row]) != targetWidth {
			return nil, 0, errors.New("composition: seam activations have inconsistent widths")
		}
		for _, value := range source[row] {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, 0, errors.New("composition: seam activations hold a non-finite value")
			}
		}
		for _, value := range target[row] {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, 0, errors.New("composition: seam activations hold a non-finite value")
			}
		}
	}
	// Normal equations: gram = SᵀS, moment = SᵀT.
	gram := make([][]float64, sourceWidth)
	moment := make([][]float64, sourceWidth)
	for i := range gram {
		gram[i] = make([]float64, sourceWidth)
		moment[i] = make([]float64, targetWidth)
	}
	targetEnergy := 0.0
	for row := range samples {
		for i := range sourceWidth {
			for j := i; j < sourceWidth; j++ {
				gram[i][j] += source[row][i] * source[row][j]
			}
			for j := range targetWidth {
				moment[i][j] += source[row][i] * target[row][j]
			}
		}
		for _, value := range target[row] {
			targetEnergy += value * value
		}
	}
	if targetEnergy == 0 {
		return nil, 0, errors.New("composition: target seam activations carry no energy")
	}
	for i := range sourceWidth {
		for j := range i {
			gram[i][j] = gram[j][i]
		}
	}
	adapter, err := solveLinearSystems(gram, moment)
	if err != nil {
		return nil, 0, err
	}
	// ||T - SW||² = ||T||² - tr(Wᵀ SᵀT) when W solves the normal equations.
	explained := 0.0
	for i := range sourceWidth {
		for j := range targetWidth {
			explained += adapter[i][j] * moment[i][j]
		}
	}
	residualEnergy := targetEnergy - explained
	if residualEnergy < 0 {
		residualEnergy = 0
	}
	return adapter, math.Sqrt(residualEnergy / targetEnergy), nil
}

// solveLinearSystems solves gram · X = moment by Gaussian elimination with
// partial pivoting; a singular gram refuses instead of inventing a map.
func solveLinearSystems(gram, moment [][]float64) ([][]float64, error) {
	n := len(gram)
	width := len(moment[0])
	augmented := make([][]float64, n)
	scale := 0.0
	for i := range augmented {
		augmented[i] = make([]float64, n+width)
		copy(augmented[i], gram[i])
		copy(augmented[i][n:], moment[i])
		if magnitude := math.Abs(gram[i][i]); magnitude > scale {
			scale = magnitude
		}
	}
	// A pivot below the accumulated float rounding floor of the gram scale
	// is numerically zero: the activations do not span the source space.
	epsilon := math.Nextafter(1, 2) - 1
	pivotFloor := scale * epsilon * float64(n)
	for column := range n {
		pivot := column
		for row := column + 1; row < n; row++ {
			if math.Abs(augmented[row][column]) > math.Abs(augmented[pivot][column]) {
				pivot = row
			}
		}
		if math.Abs(augmented[pivot][column]) <= pivotFloor {
			return nil, errors.New("composition: seam activations do not span the source space; the linear map is undetermined")
		}
		augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
		lead := augmented[column][column]
		for row := range n {
			if row == column {
				continue
			}
			factor := augmented[row][column] / lead
			if factor == 0 {
				continue
			}
			for k := column; k < n+width; k++ {
				augmented[row][k] -= factor * augmented[column][k]
			}
		}
	}
	solution := make([][]float64, n)
	for i := range solution {
		solution[i] = make([]float64, width)
		for j := range width {
			solution[i][j] = augmented[i][n+j] / augmented[i][i]
		}
	}
	return solution, nil
}

// AlignmentBiasAudit is the measured verdict on the residual proxy: over
// seams with both a computed residual and a measured post-training bridge
// parity, how often the proxy's ordering is wrong, and whether that
// violation fraction stays under the bound derived from the comparison
// count alone.
type AlignmentBiasAudit struct {
	Comparisons uint64  `json:"comparisons"`
	Violations  uint64  `json:"violations"`
	Ties        uint64  `json:"ties"`
	Bound       float64 `json:"bound"`
	Admitted    bool    `json:"admitted"`
}

// AlignmentResidualBiasAudit audits the residual proxy against measured
// bridge parity: for every strict pair of seams, a lower residual must
// come with parity at least as high — a lower-residual seam that measured
// strictly worse parity is a violation. The admissible violation fraction
// is the distribution-free bound under which a proxy with no real
// relation would pass with probability at most the registered
// denominator's reciprocal; too few comparisons make the bound
// unreachable and the audit refuses admission rather than guessing.
func AlignmentResidualBiasAudit(residuals, parities []float64) (AlignmentBiasAudit, error) {
	if len(residuals) < 2 || len(residuals) != len(parities) {
		return AlignmentBiasAudit{}, errors.New("composition: bias audit requires paired residual and parity measurements")
	}
	for index := range residuals {
		if math.IsNaN(residuals[index]) || math.IsInf(residuals[index], 0) ||
			math.IsNaN(parities[index]) || math.IsInf(parities[index], 0) {
			return AlignmentBiasAudit{}, errors.New("composition: bias audit measurements must be finite")
		}
	}
	audit := AlignmentBiasAudit{}
	for i := range len(residuals) {
		for j := i + 1; j < len(residuals); j++ {
			if residuals[i] == residuals[j] || parities[i] == parities[j] {
				audit.Ties++
				continue
			}
			audit.Comparisons++
			lowerResidual, higherResidual := i, j
			if residuals[j] < residuals[i] {
				lowerResidual, higherResidual = j, i
			}
			if parities[lowerResidual] < parities[higherResidual] {
				audit.Violations++
			}
		}
	}
	if audit.Comparisons == 0 {
		return audit, nil
	}
	audit.Bound = 0.5 - math.Sqrt(math.Log(alignmentAuditDenominator)/(2*float64(audit.Comparisons)))
	if audit.Bound > 0 &&
		float64(audit.Violations)/float64(audit.Comparisons) <= audit.Bound {
		audit.Admitted = true
	}
	return audit, nil
}
