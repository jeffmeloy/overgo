package scratchmodel

import (
	"math"
	"slices"
)

func deriveHostStage1(direction []float64, rows, cols int) int {
	var normSquared float64
	for _, value := range direction {
		normSquared += value * value
	}
	norm := math.Sqrt(normSquared)
	if norm < 1e-12 {
		return 8
	}
	dimension := min(rows, cols)
	gram := make([]float64, dimension*dimension)
	if rows <= cols {
		hostGramRows(gram, direction, rows, cols)
	} else {
		hostGramColumns(gram, direction, rows, cols)
	}
	eigenvalues := hostSymmetricEigenvalues(gram, dimension)
	minimum := slices.Min(eigenvalues)
	if minimum <= 0 {
		return 8
	}
	singular := math.Sqrt(minimum) / norm
	basin := hostCubicBasinEdge()
	iterations := 0
	for singular < basin && iterations < 12 {
		square := singular * singular
		singular = 3.4445*singular - 4.775*singular*square + 2.0315*singular*square*square
		if singular < 0 {
			singular = -singular
		}
		iterations++
	}
	return min(12, max(1, iterations+1))
}

func hostCubicBasinEdge() float64 {
	for singular := 1.0; singular > 1e-4; singular -= 1e-4 {
		if hostScalarResidual(singular, 0, 2) > 1e-2 {
			return singular + 1e-4
		}
	}
	return 1e-4
}

func hostScalarResidual(singular float64, stage1, stage2 int) float64 {
	for iteration := range stage1 + stage2 {
		coefficients := [3]float64{3.4445, -4.775, 2.0315}
		if iteration >= stage1 {
			coefficients = [3]float64{2, -1.5, 0.5}
		}
		square := singular * singular
		singular = coefficients[0]*singular + coefficients[1]*singular*square + coefficients[2]*singular*square*square
	}
	return math.Abs(singular*singular - 1)
}

func hostSymmetricEigenvalues(source []float64, size int) []float64 {
	matrix := slices.Clone(source)
	diagonal := make([]float64, size)
	offDiagonal := make([]float64, size)
	if size == 1 {
		diagonal[0] = source[0]
		return diagonal
	}
	for row := size - 1; row >= 1; row-- {
		last := row - 1
		var h, scale float64
		if last > 0 {
			for column := 0; column <= last; column++ {
				scale += math.Abs(matrix[row*size+column])
			}
			if scale == 0 {
				offDiagonal[row] = matrix[row*size+last]
			} else {
				for column := 0; column <= last; column++ {
					matrix[row*size+column] /= scale
					h += matrix[row*size+column] * matrix[row*size+column]
				}
				f := matrix[row*size+last]
				g := math.Sqrt(h)
				if f > 0 {
					g = -g
				}
				offDiagonal[row] = scale * g
				h -= f * g
				matrix[row*size+last] = f - g
				var ff float64
				for column := 0; column <= last; column++ {
					g = 0
					for inner := 0; inner <= column; inner++ {
						g += matrix[column*size+inner] * matrix[row*size+inner]
					}
					for inner := column + 1; inner <= last; inner++ {
						g += matrix[inner*size+column] * matrix[row*size+inner]
					}
					offDiagonal[column] = g / h
					ff += offDiagonal[column] * matrix[row*size+column]
				}
				half := ff / (h + h)
				for column := 0; column <= last; column++ {
					f = matrix[row*size+column]
					g = offDiagonal[column] - half*f
					offDiagonal[column] = g
					for inner := 0; inner <= column; inner++ {
						matrix[column*size+inner] -= f*offDiagonal[inner] + g*matrix[row*size+inner]
					}
				}
			}
		} else {
			offDiagonal[row] = matrix[row*size+last]
		}
		diagonal[row] = h
	}
	for index := range size {
		diagonal[index] = matrix[index*size+index]
	}
	for index := 1; index < size; index++ {
		offDiagonal[index-1] = offDiagonal[index]
	}
	offDiagonal[size-1] = 0
	for left := range size {
		for iteration := 0; iteration < 64; iteration++ {
			right := left
			for ; right < size-1; right++ {
				threshold := math.Abs(diagonal[right]) + math.Abs(diagonal[right+1])
				if math.Abs(offDiagonal[right]) <= 1e-15*threshold {
					break
				}
			}
			if right == left {
				break
			}
			g := (diagonal[left+1] - diagonal[left]) / (2 * offDiagonal[left])
			r := math.Hypot(g, 1)
			sign := r
			if g < 0 {
				sign = -r
			}
			g = diagonal[right] - diagonal[left] + offDiagonal[left]/(g+sign)
			sine, cosine, p := 1.0, 1.0, 0.0
			for index := right - 1; index >= left; index-- {
				f := sine * offDiagonal[index]
				b := cosine * offDiagonal[index]
				r = math.Hypot(f, g)
				offDiagonal[index+1] = r
				if r == 0 {
					diagonal[index+1] -= p
					offDiagonal[right] = 0
					break
				}
				sine, cosine = f/r, g/r
				g = diagonal[index+1] - p
				r = (diagonal[index]-g)*sine + 2*cosine*b
				p = sine * r
				diagonal[index+1] = g + p
				g = cosine*r - b
			}
			if r == 0 && right-1 >= left {
				continue
			}
			diagonal[left] -= p
			offDiagonal[left] = g
			offDiagonal[right] = 0
		}
	}
	return diagonal
}
