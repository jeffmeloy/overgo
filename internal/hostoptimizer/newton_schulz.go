package hostoptimizer

import (
	"math"

	"overgo/internal/hostmath"
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
	input      []float64
	transposed []float64
	gram       []float64
	square     []float64
	output     []float64
}

func (s *newtonSchulzScratch) ensure(maxMatrix, maxSquare int) {
	s.input = scratch.Resize(s.input, maxMatrix)
	s.transposed = scratch.Resize(s.transposed, maxMatrix)
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

// Every output element sums its products in one fixed order on one worker,
// so the fan-out and the contiguous read orders are bit-identical to the
// serial loops.
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
	transposed := scratch.transposed[:rows*cols]
	for iteration := range NewtonSchulzStage1Iterations + NewtonSchulzStage2Iterations {
		coefficients := NewtonSchulzStage1
		if iteration >= NewtonSchulzStage1Iterations {
			coefficients = NewtonSchulzStage2
		}
		if tall {
			transpose(transposed, input, rows, cols)
			gramRows(gram, transposed, cols, rows)
			symmetricSquare(square, gram, cols)
			polynomial(square, gram, cols, coefficients)
			matrixMultiplySymmetric(output, input, rows, cols, square)
		} else {
			gramRows(gram, input, rows, cols)
			symmetricSquare(square, gram, rows)
			polynomial(square, gram, rows, coefficients)
			matrixMultiply(output, square, rows, rows, input, cols)
		}
		copy(input, output)
	}
}

func transpose(dst, matrix []float64, rows, cols int) {
	hostmath.ParallelRangeF64(cols, rows, func(start, end int) {
		for col := start; col < end; col++ {
			for row := range rows {
				dst[col*rows+row] = matrix[row*cols+col]
			}
		}
	})
}

// gramRows fills the symmetric rows by rows Gram of a row-major matrix; a
// tall matrix passes its transpose so both operands read contiguously.
func gramRows(dst, matrix []float64, rows, cols int) {
	hostmath.ParallelRangeF64(rows, rows*cols/2, func(start, end int) {
		for upper := start; upper < end; upper++ {
			left := matrix[upper*cols : (upper+1)*cols]
			for lower := upper; lower < rows; lower++ {
				right := matrix[lower*cols : (lower+1)*cols]
				var sum float64
				for col := range cols {
					sum += left[col] * right[col]
				}
				dst[upper*rows+lower] = sum
				dst[lower*rows+upper] = sum
			}
		}
	})
}

// symmetricSquare squares a symmetric matrix; the column operand is read as
// its equal row.
func symmetricSquare(dst, matrix []float64, dimension int) {
	hostmath.ParallelRangeF64(dimension, dimension*dimension/2, func(start, end int) {
		for row := start; row < end; row++ {
			left := matrix[row*dimension : (row+1)*dimension]
			for col := row; col < dimension; col++ {
				right := matrix[col*dimension : (col+1)*dimension]
				var sum float64
				for inner := range dimension {
					sum += left[inner] * right[inner]
				}
				dst[row*dimension+col] = sum
				dst[col*dimension+row] = sum
			}
		}
	})
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

// matrixMultiplySymmetric multiplies a rows by cols matrix by a symmetric
// cols by cols matrix, reading the symmetric operand's column as its row.
func matrixMultiplySymmetric(dst, left []float64, rows, cols int, symmetric []float64) {
	hostmath.ParallelRangeF64(rows, cols*cols, func(start, end int) {
		for row := start; row < end; row++ {
			source := left[row*cols : (row+1)*cols]
			for col := range cols {
				column := symmetric[col*cols : (col+1)*cols]
				var sum float64
				for inner := range cols {
					sum += source[inner] * column[inner]
				}
				dst[row*cols+col] = sum
			}
		}
	})
}

func matrixMultiply(dst, left []float64, leftRows, shared int, right []float64, rightCols int) {
	hostmath.ParallelRangeF64(leftRows, shared*rightCols, func(start, end int) {
		for row := start; row < end; row++ {
			for col := range rightCols {
				var sum float64
				for inner := range shared {
					sum += left[row*shared+inner] * right[inner*rightCols+col]
				}
				dst[row*rightCols+col] = sum
			}
		}
	})
}
