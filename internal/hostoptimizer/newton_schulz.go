package hostoptimizer

import (
	"math"

	"overgo/internal/scratch"
)

// The two-stage Newton-Schulz iteration the host and device orthogonalizers
// share: stage 1 iterations, stage 2 iterations, the Frobenius norm below
// which a matrix is left untouched, and the polynomial coefficients.
const (
	// NewtonSchulzStage1Iterations counts the aggressive first-stage steps.
	NewtonSchulzStage1Iterations = 8
	// NewtonSchulzStage2Iterations counts the refining second-stage steps.
	NewtonSchulzStage2Iterations = 2
	// NewtonSchulzFrobeniusGuard is the norm floor for a nonzero matrix.
	NewtonSchulzFrobeniusGuard = 1e-12
)

var (
	// NewtonSchulzStage1 holds the first-stage polynomial coefficients.
	NewtonSchulzStage1 = [3]float64{3.4445, -4.7750, 2.0315}
	// NewtonSchulzStage2 holds the second-stage polynomial coefficients.
	NewtonSchulzStage2 = [3]float64{2, -1.5, 0.5}
)

type newtonSchulzScratch struct {
	input  []float64
	gram   []float64
	square []float64
	output []float64
}

func (s *newtonSchulzScratch) ensure(maxMatrix, maxSquare int) {
	s.input = scratch.Resize(s.input, maxMatrix)
	s.gram = scratch.Resize(s.gram, maxSquare)
	s.square = scratch.Resize(s.square, maxSquare)
	s.output = scratch.Resize(s.output, maxMatrix)
}

// NewtonSchulz orthogonalizes a rows by cols matrix in place with fresh
// scratch; the device stepper's parity check compares against it.
func NewtonSchulz(input []float64, rows, cols int) {
	var scratch newtonSchulzScratch
	square := min(rows, cols)
	scratch.ensure(rows*cols, square*square)
	newtonSchulz(input, rows, cols, &scratch)
}

func newtonSchulz(input []float64, rows, cols int, scratch *newtonSchulzScratch) {
	var normSquared float64
	for _, value := range input {
		normSquared += value * value
	}
	norm := math.Sqrt(normSquared)
	if norm < NewtonSchulzFrobeniusGuard {
		return
	}
	for index := range input {
		input[index] /= norm
	}

	tall := rows >= cols
	dimension := min(rows, cols)
	gram := scratch.gram[:dimension*dimension]
	square := scratch.square[:dimension*dimension]
	output := scratch.output[:rows*cols]
	for iteration := range NewtonSchulzStage1Iterations + NewtonSchulzStage2Iterations {
		coefficients := NewtonSchulzStage1
		if iteration >= NewtonSchulzStage1Iterations {
			coefficients = NewtonSchulzStage2
		}
		if tall {
			gramColumns(gram, input, rows, cols)
			symmetricSquare(square, gram, cols)
			polynomial(square, gram, cols, coefficients)
			matrixMultiply(output, input, rows, cols, square, cols)
		} else {
			gramRows(gram, input, rows, cols)
			symmetricSquare(square, gram, rows)
			polynomial(square, gram, rows, coefficients)
			matrixMultiply(output, square, rows, rows, input, cols)
		}
		copy(input, output)
	}
}

func gramColumns(dst, matrix []float64, rows, cols int) {
	for left := range cols {
		for right := left; right < cols; right++ {
			var sum float64
			for row := range rows {
				sum += matrix[row*cols+left] * matrix[row*cols+right]
			}
			dst[left*cols+right] = sum
			dst[right*cols+left] = sum
		}
	}
}

func gramRows(dst, matrix []float64, rows, cols int) {
	for upper := range rows {
		for lower := upper; lower < rows; lower++ {
			var sum float64
			for col := range cols {
				sum += matrix[upper*cols+col] * matrix[lower*cols+col]
			}
			dst[upper*rows+lower] = sum
			dst[lower*rows+upper] = sum
		}
	}
}

func symmetricSquare(dst, matrix []float64, dimension int) {
	for row := range dimension {
		for col := row; col < dimension; col++ {
			var sum float64
			for inner := range dimension {
				sum += matrix[row*dimension+inner] * matrix[inner*dimension+col]
			}
			dst[row*dimension+col] = sum
			dst[col*dimension+row] = sum
		}
	}
}

func polynomial(dst, gram []float64, dimension int, coefficients [3]float64) {
	for row := range dimension {
		for col := range dimension {
			index := row*dimension + col
			value := coefficients[1]*gram[index] + coefficients[2]*dst[index]
			if row == col {
				value += coefficients[0]
			}
			dst[index] = value
		}
	}
}

func matrixMultiply(dst, left []float64, leftRows, shared int, right []float64, rightCols int) {
	for row := range leftRows {
		for col := range rightCols {
			var sum float64
			for inner := range shared {
				sum += left[row*shared+inner] * right[inner*rightCols+col]
			}
			dst[row*rightCols+col] = sum
		}
	}
}
