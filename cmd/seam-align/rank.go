package main

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"overgo/internal/composition"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
)

// seamFolds are the folds a held-out residual divides its samples' groups among, and seamDimensionName the score
// dimension a realized seam's measured residual stands in.
const (
	seamFolds         = 5
	seamDimensionName = "seam-residual"
)

// sourcePrefix names a captured source representation's tensor; the capture's target is targetTensor and the source
// video each sample comes from groupsTensor.
const (
	sourcePrefix = "source/"
	targetTensor = "target"
	groupsTensor = "groups"
)

// seamRank is one captured source's measured seam: its width, the residual fitted over every sample, the residual on
// samples held out of each fit, the same with each sample's target taken from another group, and the rung its held-out
// residual reaches.
type seamRank struct {
	Source    string                  `json:"source"`
	Width     int                     `json:"width"`
	Samples   int                     `json:"samples"`
	InSample  float64                 `json:"in_sample"`
	HeldOut   float64                 `json:"held_out"`
	Unrelated float64                 `json:"unrelated"`
	Rung      composition.AdapterRung `json:"rung"`
}

// rankSeams prints a capture's ranked seams, best first.
func rankSeams(path string) error {
	ranks, err := rankCapture(path)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(ranks)
}

// rankCapture ranks each source representation a capture holds by its held-out residual against the capture's target,
// beside the residual of a pairing that holds no relation, best first: a seam is measured only where its held-out
// residual stands well below its unrelated one.
func rankCapture(path string) ([]seamRank, error) {
	capture, err := safetensors.OpenFile(path)
	if err != nil {
		return nil, err
	}
	defer capture.Close()
	target, err := captureRows(capture, targetTensor)
	if err != nil {
		return nil, err
	}
	groupRows, err := captureRows(capture, groupsTensor)
	if err != nil {
		return nil, err
	}
	groups := make([]int, len(groupRows))
	for index, row := range groupRows {
		groups[index] = int(row[0])
	}
	var ranks []seamRank
	for _, name := range capture.Names() {
		label, isSource := strings.CutPrefix(name, sourcePrefix)
		if !isSource {
			continue
		}
		source, err := captureRows(capture, name)
		if err != nil {
			return nil, err
		}
		rank, err := measureSeam(label, source, target, groups)
		if err != nil {
			return nil, err
		}
		ranks = append(ranks, rank)
	}
	if len(ranks) == 0 {
		return nil, errors.New("seam-align: the capture holds no source representation")
	}
	slices.SortFunc(ranks, func(left, right seamRank) int { return cmp.Compare(left.HeldOut, right.HeldOut) })
	return ranks, nil
}

// measureSeam measures one source representation's seam against the target over its samples.
func measureSeam(label string, source, target [][]float64, groups []int) (seamRank, error) {
	inSample, err := composition.SeamAlignmentResidual(source, target)
	if err != nil {
		return seamRank{}, fmt.Errorf("seam-align: %s: %w", label, err)
	}
	heldOut, unrelated, err := heldOutResiduals(source, target, groups)
	if err != nil {
		return seamRank{}, fmt.Errorf("seam-align: %s: %w", label, err)
	}
	return seamRank{
		Source: label, Width: len(source[0]), Samples: len(source), InSample: inSample, HeldOut: heldOut, Unrelated: unrelated,
		Rung: composition.AdapterRungFor(heldOut),
	}, nil
}

// heldOutResiduals are a seam's held-out residual and that of the same representations paired across groups.
func heldOutResiduals(source, target [][]float64, groups []int) (heldOut, unrelated float64, err error) {
	if heldOut, err = composition.HeldOutSeamResidual(source, target, groups, seamFolds); err != nil {
		return 0, 0, err
	}
	pairing, err := composition.UnrelatedPairing(groups)
	if err != nil {
		return 0, 0, err
	}
	paired := make([][]float64, len(target))
	for sample, other := range pairing {
		paired[sample] = target[other]
	}
	unrelated, err = composition.HeldOutSeamResidual(source, paired, groups, seamFolds)
	return heldOut, unrelated, err
}

// seamDimension is the score dimension a seam's measured residuals give: its held-out residual against the largest the
// linear rung admits, the lower better, so a seam improves only where a linear adapter explains more than half the
// held-out target. Each sample stands for a group of its own.
func seamDimension(source, target [][]float64) (composition.CompositeDimension, error) {
	groups := make([]int, len(source))
	for index := range groups {
		groups[index] = index
	}
	heldOut, err := composition.HeldOutSeamResidual(source, target, groups, seamFolds)
	if err != nil {
		return composition.CompositeDimension{}, err
	}
	return composition.CompositeDimension{Baseline: composition.LinearRungResidual(), Candidate: heldOut, Direction: runrecord.DirectionMinimize}, nil
}

// matrixRank is a matrix's dimensions.
const matrixRank = 2

// captureRows reads one captured tensor as rows of float64: a matrix's rows, or a vector's values a row each.
func captureRows(capture *safetensors.Source, name string) ([][]float64, error) {
	tensor, found := capture.Tensors[name]
	if !found {
		return nil, fmt.Errorf("seam-align: the capture holds no %s tensor", name)
	}
	values, err := safetensors.ReadF32(tensor)
	if err != nil {
		return nil, err
	}
	width := 1
	if len(tensor.Shape) == matrixRank {
		width = int(tensor.Shape[1])
	} else if len(tensor.Shape) != 1 {
		return nil, fmt.Errorf("seam-align: tensor %s is neither a vector nor a matrix", name)
	}
	rows := make([][]float64, len(values)/width)
	for row := range rows {
		rows[row] = make([]float64, width)
		for column := range width {
			rows[row][column] = float64(values[row*width+column])
		}
	}
	return rows, nil
}
