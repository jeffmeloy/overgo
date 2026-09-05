package optimizer

import (
	"math"

	"overgo/internal/scratch"
)

const (
	newtonSchulzStage1Iterations = 8
	newtonSchulzStage2Iterations = 2
	newtonSchulzFrobeniusGuard   = 1e-12
)

var (
	newtonSchulzStage1 = [3]float64{3.4445, -4.7750, 2.0315}
	newtonSchulzStage2 = [3]float64{2, -1.5, 0.5}
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

func newtonSchulz(input []float64, rows, cols int, scratch *newtonSchulzScratch) {
	var normSquared float64
	for _, value := range input {
		normSquared += value * value
	}
	norm := math.Sqrt(normSquared)
	if norm < newtonSchulzFrobeniusGuard {
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
	for iteration := range newtonSchulzStage1Iterations + newtonSchulzStage2Iterations {
		coefficients := newtonSchulzStage1
		if iteration >= newtonSchulzStage1Iterations {
			coefficients = newtonSchulzStage2
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
