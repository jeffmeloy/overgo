package scratchmodel

import (
	"errors"
	"math"
	"slices"
)

const hostMuonMomentum = 29.0 / 31.0

type HostTrainingResult struct {
	Losses         []float64
	ValidationLoss float64
	Checkpoint     HostCheckpoint
}

type HostCheckpoint struct {
	Model      HostModelState
	Momentum   map[string][]float64
	NSStage1   map[string]int
	Step       int
	TotalSteps int
	Document   int
	ProgramID  string
}

type HostModelState struct {
	Weights map[string][]float64
}

func (c Construction) MuonProbe(documents []string) ([]HostGroup, error) {
	probe, err := c.Probe(documents)
	if err != nil {
		return nil, err
	}
	momentum := make(map[string][]float64, len(probe.Groups))
	groups := make([]HostGroup, len(probe.Groups))
	for index, group := range probe.Groups {
		groups[index] = group
		momentum[group.Name] = make([]float64, len(group.Weights))
		before := slices.Clone(group.Weights)
		muonUpdate(group.Weights, group.Gradients, momentum[group.Name], group.Rows, group.Cols, 1, hostMuonMomentum, 8)
		for offset := range group.Weights {
			groups[index].Gradients[offset] = before[offset] - group.Weights[offset]
		}
	}
	return groups, nil
}

func (c Construction) TrainHost(totalSteps, steps int, resume *HostCheckpoint) (HostTrainingResult, error) {
	if totalSteps <= 0 || steps <= 0 || steps > totalSteps {
		return HostTrainingResult{}, errors.New("scratch model: training steps must be positive")
	}
	state := c.hostState()
	momentum := make(map[string][]float64, len(c.parameters))
	stages := make(map[string]int, len(c.parameters))
	startStep, documentCursor := 0, 0
	if resume != nil {
		if resume.ProgramID != c.authority.ID().String() || resume.TotalSteps != totalSteps ||
			resume.Step < 0 || resume.Step+steps > totalSteps || resume.Document != resume.Step {
			return HostTrainingResult{}, errors.New("scratch model: checkpoint authority differs")
		}
		if err := restoreHostState(state, resume.Model.Weights); err != nil {
			return HostTrainingResult{}, err
		}
		if len(resume.Momentum) != len(c.parameters) || len(resume.NSStage1) != len(c.parameters) {
			return HostTrainingResult{}, errors.New("scratch model: checkpoint optimizer count differs")
		}
		for _, parameter := range c.parameters {
			values, ok := resume.Momentum[parameter.Name]
			if !ok || len(values) != parameter.Rows*parameter.Cols || !allFinite(values) {
				return HostTrainingResult{}, errors.New("scratch model: checkpoint momentum differs")
			}
			stage, ok := resume.NSStage1[parameter.Name]
			if !ok || min(parameter.Rows, parameter.Cols) < 2 && stage != 0 ||
				min(parameter.Rows, parameter.Cols) >= 2 && (stage < 1 || stage > 12) {
				return HostTrainingResult{}, errors.New("scratch model: checkpoint Muon stage differs")
			}
			momentum[parameter.Name] = slices.Clone(values)
			stages[parameter.Name] = stage
		}
		startStep, documentCursor = resume.Step, resume.Document
	}
	for _, parameter := range c.parameters {
		if momentum[parameter.Name] == nil {
			momentum[parameter.Name] = make([]float64, parameter.Rows*parameter.Cols)
		}
	}
	model := hostModel{config: c.config, state: state}
	losses := make([]float64, steps)
	for localStep := range steps {
		document := c.split.Train[documentCursor%len(c.split.Train)]
		documentCursor++
		tokens, err := c.Tokens(document)
		if err != nil {
			return HostTrainingResult{}, err
		}
		losses[localStep], _ = model.document(tokens, 1)
		step := startStep + localStep + 1
		rate := c.config.BaseLR * max(0, 1-float64(step)/float64(totalSteps))
		for _, parameter := range c.parameters {
			matrix := state[parameter.Name]
			weights := flattenValueData(matrix)
			gradients := flattenValueGrad(matrix)
			stages[parameter.Name] = muonUpdate(
				weights, gradients, momentum[parameter.Name], parameter.Rows, parameter.Cols,
				rate, hostMuonMomentum, stages[parameter.Name],
			)
			writeHostValues(matrix, weights, true)
		}
	}
	validationTokens, err := c.Tokens(c.split.Validation[0])
	if err != nil {
		return HostTrainingResult{}, err
	}
	validationLoss, _ := model.document(validationTokens, 0)
	checkpoint := HostCheckpoint{
		Model: HostModelState{Weights: snapshotHostState(state)}, Momentum: cloneWeights(momentum), NSStage1: cloneIntMap(stages),
		Step: startStep + steps, TotalSteps: totalSteps, Document: documentCursor, ProgramID: c.authority.ID().String(),
	}
	return HostTrainingResult{Losses: losses, ValidationLoss: validationLoss, Checkpoint: checkpoint}, nil
}

func muonUpdate(weights, gradients, momentum []float64, rows, cols int, rate, mu float64, stage1 int) int {
	direction := make([]float64, len(weights))
	for index, gradient := range gradients {
		momentum[index] = mu*momentum[index] + gradient
		direction[index] = mu*momentum[index] + gradient
	}
	if min(rows, cols) < 2 {
		for index, gradient := range gradients {
			if gradient == 0 {
				continue
			}
			if direction[index] > 0 {
				weights[index] -= rate
			} else if direction[index] < 0 {
				weights[index] += rate
			}
		}
		return 0
	}
	if stage1 <= 0 {
		stage1 = deriveHostStage1(direction, rows, cols)
	}
	hostNewtonSchulz(direction, rows, cols, stage1, 2)
	scale := math.Sqrt(float64(max(rows, cols))) * math.Sqrt(1-mu*mu) * rate
	for index := range weights {
		weights[index] -= scale * direction[index]
	}
	return stage1
}

func hostNewtonSchulz(matrix []float64, rows, cols, stage1, stage2 int) {
	var normSquared float64
	for _, value := range matrix {
		normSquared += value * value
	}
	norm := math.Sqrt(normSquared)
	if norm < 1e-12 {
		return
	}
	for index := range matrix {
		matrix[index] /= norm
	}
	dimension := min(rows, cols)
	gram := make([]float64, dimension*dimension)
	square := make([]float64, dimension*dimension)
	output := make([]float64, rows*cols)
	for iteration := range stage1 + stage2 {
		coefficients := [3]float64{3.4445, -4.775, 2.0315}
		if iteration >= stage1 {
			coefficients = [3]float64{2, -1.5, 0.5}
		}
		if rows >= cols {
			hostGramColumns(gram, matrix, rows, cols)
			hostSymmetricSquare(square, gram, cols)
			hostPolynomial(square, gram, cols, coefficients)
			clear(output)
			matrixMultiply(output, matrix, rows, cols, square, cols)
		} else {
			hostGramRows(gram, matrix, rows, cols)
			hostSymmetricSquare(square, gram, rows)
			hostPolynomial(square, gram, rows, coefficients)
			clear(output)
			matrixMultiply(output, square, rows, rows, matrix, cols)
		}
		copy(matrix, output)
	}
}

func hostGramColumns(dst, matrix []float64, rows, cols int) {
	for left := range cols {
		for right := left; right < cols; right++ {
			var sum float64
			for row := range rows {
				sum += matrix[row*cols+left] * matrix[row*cols+right]
			}
			dst[left*cols+right], dst[right*cols+left] = sum, sum
		}
	}
}

func hostGramRows(dst, matrix []float64, rows, cols int) {
	for upper := range rows {
		for lower := upper; lower < rows; lower++ {
			var sum float64
			for column := range cols {
				sum += matrix[upper*cols+column] * matrix[lower*cols+column]
			}
			dst[upper*rows+lower], dst[lower*rows+upper] = sum, sum
		}
	}
}

func hostSymmetricSquare(dst, matrix []float64, size int) {
	for row := range size {
		for column := row; column < size; column++ {
			var sum float64
			for inner := range size {
				sum += matrix[row*size+inner] * matrix[inner*size+column]
			}
			dst[row*size+column], dst[column*size+row] = sum, sum
		}
	}
}

func hostPolynomial(dst, gram []float64, size int, coefficients [3]float64) {
	for row := range size {
		for column := range size {
			index := row*size + column
			value := coefficients[1]*gram[index] + coefficients[2]*dst[index]
			if row == column {
				value += coefficients[0]
			}
			dst[index] = value
		}
	}
}

func snapshotHostState(state hostState) map[string][]float64 {
	result := make(map[string][]float64, len(state))
	for name, matrix := range state {
		result[name] = flattenValueData(matrix)
	}
	return result
}

func restoreHostState(state hostState, weights map[string][]float64) error {
	if len(weights) != len(state) {
		return errors.New("scratch model: checkpoint tensor count differs")
	}
	for name, matrix := range state {
		values, ok := weights[name]
		if !ok || len(values) != len(matrix)*len(matrix[0]) || !allFinite(values) {
			return errors.New("scratch model: checkpoint tensor differs")
		}
		writeHostValues(matrix, values, false)
	}
	return nil
}

func allFinite(values []float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func writeHostValues(matrix [][]*value, values []float64, clearGradient bool) {
	columns := len(matrix[0])
	for row := range matrix {
		for column := range matrix[row] {
			item := matrix[row][column]
			item.data = values[row*columns+column]
			if clearGradient {
				item.grad = 0
			}
		}
	}
}

func cloneIntMap(source map[string]int) map[string]int {
	result := make(map[string]int, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
