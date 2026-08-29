package hostmath

import (
	"errors"
	"math"
)

// SelectedLogProbInto writes selected row log probabilities from x*w^T+b.
func SelectedLogProbInto(
	dst []float64,
	x, weight, bias []float32,
	targets []int,
	selected []bool,
	rows, input, output int,
	workspace []float32,
) error {
	if err := validateSelectedProjection(len(dst), x, weight, bias, targets, selected, rows, input, output, workspace); err != nil {
		return err
	}
	for row := range rows {
		if !selected[row] {
			dst[row] = 0
			continue
		}
		logZ, target := projectionLogZ(x[row*input:(row+1)*input], weight, bias, targets[row], input, output, workspace)
		dst[row] = float64(target) - logZ
	}
	return nil
}

// SelectedLogProbBackward applies selected-log-probability VJPs in place.
func SelectedLogProbBackward(
	dX, dWeight, dBias []float32,
	x, weight, bias []float32,
	targets []int,
	selected []bool,
	dLogProb []float64,
	rows, input, output int,
	workspace []float32,
	accumulate bool,
) error {
	if err := validateSelectedProjection(rows, x, weight, bias, targets, selected, rows, input, output, workspace); err != nil {
		return err
	}
	if len(dX) != rows*input || len(dWeight) != 0 && len(dWeight) != output*input ||
		len(dLogProb) != rows || len(dBias) != 0 && len(dBias) != output {
		return errors.New("hostmath: selected log-probability gradient extent differs")
	}
	if !accumulate {
		clear(dX)
		clear(dWeight)
		clear(dBias)
	}
	for row := range rows {
		if !selected[row] || dLogProb[row] == 0 {
			continue
		}
		xRow := x[row*input : (row+1)*input]
		dXRow := dX[row*input : (row+1)*input]
		logZ, _ := projectionLogZ(xRow, weight, bias, targets[row], input, output, workspace)
		for start := 0; start < output; start += len(workspace) {
			end := min(start+len(workspace), output)
			projectionTile(workspace[:end-start], xRow, weight, bias, start, input)
			for local, logit := range workspace[:end-start] {
				column := start + local
				gradient := -dLogProb[row] * math.Exp(float64(logit)-logZ)
				if column == targets[row] {
					gradient += dLogProb[row]
				}
				g := float32(gradient)
				if len(dBias) != 0 {
					dBias[column] += g
				}
				wRow := weight[column*input : (column+1)*input]
				if dWeight == nil {
					for index := range xRow {
						dXRow[index] += g * wRow[index]
					}
					continue
				}
				dWRow := dWeight[column*input : (column+1)*input]
				for index, value := range xRow {
					dWRow[index] += g * value
					dXRow[index] += g * wRow[index]
				}
			}
		}
	}
	return nil
}

func projectionLogZ(x, weight, bias []float32, target, input, output int, workspace []float32) (float64, float32) {
	maximum := float32(-math.MaxFloat32)
	var targetLogit float32
	for start := 0; start < output; start += len(workspace) {
		end := min(start+len(workspace), output)
		tile := workspace[:end-start]
		projectionTile(tile, x, weight, bias, start, input)
		for local, logit := range tile {
			maximum = max(maximum, logit)
			if start+local == target {
				targetLogit = logit
			}
		}
	}
	var sum float64
	for start := 0; start < output; start += len(workspace) {
		end := min(start+len(workspace), output)
		tile := workspace[:end-start]
		projectionTile(tile, x, weight, bias, start, input)
		for _, logit := range tile {
			sum += math.Exp(float64(logit - maximum))
		}
	}
	return float64(maximum) + math.Log(sum), targetLogit
}

func projectionTile(dst, x, weight, bias []float32, start, input int) {
	for local := range dst {
		column := start + local
		wRow := weight[column*input : (column+1)*input]
		var value float32
		for index := range x {
			value += x[index] * wRow[index]
		}
		if len(bias) != 0 {
			value += bias[column]
		}
		dst[local] = value
	}
}

func validateSelectedProjection(
	destinationRows int,
	x, weight, bias []float32,
	targets []int,
	selected []bool,
	rows, input, output int,
	workspace []float32,
) error {
	maxInt := int(^uint(0) >> 1)
	if rows <= 0 || input <= 0 || output <= 0 || rows > maxInt/input || output > maxInt/input ||
		destinationRows != rows || len(x) != rows*input || len(weight) != output*input ||
		len(bias) != 0 && len(bias) != output || len(targets) != rows || len(selected) != rows ||
		len(workspace) == 0 {
		return errors.New("hostmath: selected log-probability extent differs")
	}
	for row, target := range targets {
		if selected[row] && (target < 0 || target >= output) {
			return errors.New("hostmath: selected log-probability target differs")
		}
	}
	return nil
}
