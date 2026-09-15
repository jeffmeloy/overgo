package hostoptimizer

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// serialNewtonSchulz is the orthogonalizer's original serial form: the
// fan-out and the contiguous read orders must reproduce it bit for bit.
func serialNewtonSchulz(input []float64, rows, cols int) {
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
	gram := make([]float64, dimension*dimension)
	square := make([]float64, dimension*dimension)
	output := make([]float64, rows*cols)
	for iteration := range NewtonSchulzStage1Iterations + NewtonSchulzStage2Iterations {
		coefficients := NewtonSchulzStage1
		if iteration >= NewtonSchulzStage1Iterations {
			coefficients = NewtonSchulzStage2
		}
		if tall {
			for left := range cols {
				for right := left; right < cols; right++ {
					var sum float64
					for row := range rows {
						sum += input[row*cols+left] * input[row*cols+right]
					}
					gram[left*cols+right] = sum
					gram[right*cols+left] = sum
				}
			}
		} else {
			for upper := range rows {
				for lower := upper; lower < rows; lower++ {
					var sum float64
					for col := range cols {
						sum += input[upper*cols+col] * input[lower*cols+col]
					}
					gram[upper*rows+lower] = sum
					gram[lower*rows+upper] = sum
				}
			}
		}
		for row := range dimension {
			for col := row; col < dimension; col++ {
				var sum float64
				for inner := range dimension {
					sum += gram[row*dimension+inner] * gram[inner*dimension+col]
				}
				square[row*dimension+col] = sum
				square[col*dimension+row] = sum
			}
		}
		polynomial(square, gram, dimension, coefficients)
		if tall {
			for row := range rows {
				for col := range cols {
					var sum float64
					for inner := range cols {
						sum += input[row*cols+inner] * square[inner*cols+col]
					}
					output[row*cols+col] = sum
				}
			}
		} else {
			for row := range rows {
				for col := range cols {
					var sum float64
					for inner := range rows {
						sum += square[row*rows+inner] * input[inner*cols+col]
					}
					output[row*cols+col] = sum
				}
			}
		}
		copy(input, output)
	}
}

// TestNewtonSchulzMatchesSerialForm holds the parallel orthogonalizer to the
// serial loops bit for bit on tall, wide and square matrices, and holds the
// result orthogonal.
func TestNewtonSchulzMatchesSerialForm(t *testing.T) {
	t.Parallel()
	for _, shape := range []struct{ rows, cols int }{{257, 48}, {48, 257}, {96, 96}, {3, 1}, {1, 3}} {
		generator := rand.New(rand.NewPCG(uint64(shape.rows), uint64(shape.cols)))
		input := make([]float64, shape.rows*shape.cols)
		for index := range input {
			input[index] = generator.NormFloat64()
		}
		reference := slices.Clone(input)
		serialNewtonSchulz(reference, shape.rows, shape.cols)
		NewtonSchulz(input, shape.rows, shape.cols)
		if !slices.Equal(input, reference) {
			t.Fatalf("%dx%d: parallel orthogonalizer differs from the serial form", shape.rows, shape.cols)
		}
		dimension := min(shape.rows, shape.cols)
		tall := shape.rows >= shape.cols
		for left := range dimension {
			for right := range dimension {
				var sum float64
				if tall {
					for row := range shape.rows {
						sum += input[row*shape.cols+left] * input[row*shape.cols+right]
					}
				} else {
					for col := range shape.cols {
						sum += input[left*shape.cols+col] * input[right*shape.cols+col]
					}
				}
				want := 0.0
				if left == right {
					want = 1
				}
				if math.Abs(sum-want) > 0.5 {
					t.Fatalf("%dx%d: Gram[%d,%d]=%g is not orthonormal", shape.rows, shape.cols, left, right, sum)
				}
			}
		}
	}
	zero := make([]float64, 6)
	NewtonSchulz(zero, 2, 3)
	if slices.ContainsFunc(zero, func(value float64) bool { return value != 0 }) {
		t.Fatal("a matrix under the Frobenius guard changed")
	}
}
