package composition

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// minimumSeamFolds are the folds a held-out residual needs: one to fit on and one to hold out.
const minimumSeamFolds = 2

// HeldOutSeamResidual measures how much of the target seam representation a linear adapter cannot reproduce on
// samples it was not fitted to. The samples fall into folds by group, a group wholly in one fold; each fold's adapter is
// fitted over the other folds with the target centered on them, and the residual is the held-out folds' unexplained
// share of their centered target energy. A pairing that shares nothing linear reads near one however many samples it
// has, where the in-sample residual reads lower the fewer samples there are, and a target's shared mean explains
// nothing; the source is not centered, since one taken from a normalization carries its constant in its span.
func HeldOutSeamResidual(source, target [][]float64, groups []int, folds int) (float64, error) {
	if len(source) != len(target) || len(groups) != len(source) || folds < minimumSeamFolds {
		return 0, errors.New("composition: a held-out residual requires paired activations, a group each and two folds")
	}
	var unexplained, energy float64
	for fold := range folds {
		var fitSource, fitTarget, heldSource, heldTarget [][]float64
		for sample, group := range groups {
			if group%folds == fold {
				heldSource, heldTarget = append(heldSource, source[sample]), append(heldTarget, target[sample])
			} else {
				fitSource, fitTarget = append(fitSource, source[sample]), append(fitTarget, target[sample])
			}
		}
		if len(heldSource) == 0 || len(fitSource) == 0 {
			return 0, fmt.Errorf("composition: fold %d of %d holds no samples or all of them", fold, folds)
		}
		mean := make([]float64, len(fitTarget[0]))
		for _, row := range fitTarget {
			for column, value := range row {
				mean[column] += value / float64(len(fitTarget))
			}
		}
		adapter, _, err := AlignSeamAdapter(fitSource, centered(fitTarget, mean))
		if err != nil {
			return 0, fmt.Errorf("composition: fold %d: %w", fold, err)
		}
		for sample, row := range heldSource {
			for column := range mean {
				var predicted float64
				for index, value := range row {
					predicted += value * adapter[index][column]
				}
				deviation := heldTarget[sample][column] - mean[column]
				unexplained += (deviation - predicted) * (deviation - predicted)
				energy += deviation * deviation
			}
		}
	}
	if energy == 0 {
		return 0, errors.New("composition: held-out target activations do not vary")
	}
	return math.Sqrt(unexplained / energy), nil
}

// centered is rows less mean, each in a row of its own.
func centered(rows [][]float64, mean []float64) [][]float64 {
	result := make([][]float64, len(rows))
	for index, row := range rows {
		result[index] = make([]float64, len(row))
		for column, value := range row {
			result[index][column] = value - mean[column]
		}
	}
	return result
}

// UnrelatedPairing pairs each sample with another group's: the samples ordered by group, each takes the sample as many
// places on as the largest group holds, so no sample keeps its own group's target and every target is taken once. It
// is the planted pairing a measured seam must beat, of the same representations with their relation removed.
func UnrelatedPairing(groups []int) ([]int, error) {
	order := make([]int, len(groups))
	for index := range order {
		order[index] = index
	}
	slices.SortStableFunc(order, func(left, right int) int { return groups[left] - groups[right] })
	sizes := map[int]int{}
	largest := 0
	for _, group := range groups {
		sizes[group]++
		largest = max(largest, sizes[group])
	}
	if largest > len(groups)-largest {
		return nil, errors.New("composition: an unrelated pairing needs groups none of which holds half the samples")
	}
	pairing := make([]int, len(groups))
	for place, sample := range order {
		pairing[sample] = order[(place+largest)%len(order)]
	}
	return pairing, nil
}
