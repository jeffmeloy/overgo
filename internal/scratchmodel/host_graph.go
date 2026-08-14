package scratchmodel

import (
	"math"
	"slices"
)

type value struct {
	data float64
	grad float64
}

type tape struct{ backward []func() }

func (t *tape) node(data float64) *value { return &value{data: data} }
func (t *tape) append(backward func())   { t.backward = append(t.backward, backward) }
func (t *tape) run(root *value) {
	root.grad = 1
	for index := len(t.backward) - 1; index >= 0; index-- {
		t.backward[index]()
	}
}

type rankedValue struct {
	value float64
	index int
}

type rankedDistance struct {
	distance float64
	index    int
}

func rankWeights(size int) (normalized, raw []float64) {
	normalized = make([]float64, size)
	raw = make([]float64, size)
	for index := range size {
		raw[index] = 2*(float64(index)+0.5)/float64(size) - 1
		normalized[index] = raw[index] / float64(size)
	}
	return normalized, raw
}

func madnormForward(input, normalizedWeights, rawWeights []float64, epsilon float64) (output, a, b []float64, c, inverseScale float64) {
	size := len(input)
	ranked := make([]rankedValue, size)
	for index, item := range input {
		ranked[index] = rankedValue{value: item, index: index}
	}
	slices.SortFunc(ranked, func(left, right rankedValue) int {
		if left.value < right.value {
			return -1
		}
		if left.value > right.value {
			return 1
		}
		return 0
	})
	var location float64
	for index, item := range ranked {
		location += item.value * normalizedWeights[index]
	}
	split := 0
	for split < size && ranked[split].value < location {
		split++
	}
	distances := make([]rankedDistance, size)
	left, right := split-1, split
	for index := range size {
		switch {
		case left < 0:
			distances[index] = rankedDistance{ranked[right].value - location, right}
			right++
		case right >= size:
			distances[index] = rankedDistance{location - ranked[left].value, left}
			left--
		default:
			leftDistance := location - ranked[left].value
			rightDistance := ranked[right].value - location
			if leftDistance <= rightDistance {
				distances[index] = rankedDistance{leftDistance, left}
				left--
			} else {
				distances[index] = rankedDistance{rightDistance, right}
				right++
			}
		}
	}
	twoOverSize := 2 / float64(size)
	var scale float64
	for index, distance := range distances {
		scale += distance.distance * rawWeights[index]
	}
	inverseScale = 1 / (scale*twoOverSize + epsilon)
	output = make([]float64, size)
	for index := range size {
		output[index] = (input[index] - location) * inverseScale
	}
	distanceRank := make([]int, size)
	for index, distance := range distances {
		distanceRank[distance.index] = index
	}
	boundary := make([]float64, size)
	for index := range size {
		if ranked[index].value >= location {
			boundary[index] = 1
		} else {
			boundary[index] = -1
		}
	}
	bc := make([]float64, size)
	for index := range size {
		bc[index] = rawWeights[distanceRank[index]] * twoOverSize * boundary[index]
		c += bc[index]
	}
	originalRank := make([]int, size)
	for index, item := range ranked {
		originalRank[item.index] = index
	}
	a, b = make([]float64, size), make([]float64, size)
	for index := range size {
		a[index] = normalizedWeights[originalRank[index]]
		b[index] = bc[originalRank[index]]
	}
	return output, a, b, c, inverseScale
}

func madnormBackward(incoming, output, a, b []float64, c, inverseScale float64) []float64 {
	var sum, weightedSum float64
	for index := range incoming {
		sum += incoming[index]
		weightedSum += incoming[index] * output[index]
	}
	t1 := inverseScale * (sum + c*weightedSum)
	t2 := inverseScale * weightedSum
	result := make([]float64, len(incoming))
	for index := range result {
		result[index] = inverseScale*incoming[index] - a[index]*t1 - b[index]*t2
	}
	return result
}

func normalize(t *tape, input []*value, normalizedWeights, rawWeights []float64, epsilon float64) []*value {
	data := make([]float64, len(input))
	for index, item := range input {
		data[index] = item.data
	}
	output, a, b, c, inverseScale := madnormForward(data, normalizedWeights, rawWeights, epsilon)
	result := make([]*value, len(output))
	for index, item := range output {
		result[index] = t.node(item)
	}
	t.append(func() {
		incoming := make([]float64, len(result))
		for index, item := range result {
			incoming[index] = item.grad
		}
		gradient := madnormBackward(incoming, output, a, b, c, inverseScale)
		for index := range input {
			input[index].grad += gradient[index]
		}
	})
	return result
}

func linear(t *tape, input []*value, weights [][]*value, residual []*value) []*value {
	result := make([]*value, len(weights))
	for row, matrixRow := range weights {
		var sum float64
		for column := range input {
			sum += matrixRow[column].data * input[column].data
		}
		if residual != nil {
			sum += residual[row].data
		}
		result[row] = t.node(sum)
	}
	t.append(func() {
		inputGradient := make([]float64, len(input))
		for row, matrixRow := range weights {
			gradient := result[row].grad
			if gradient != 0 {
				for column := range input {
					inputGradient[column] += matrixRow[column].data * gradient
					matrixRow[column].grad += gradient * input[column].data
				}
			}
		}
		for column := range input {
			input[column].grad += inputGradient[column]
		}
		for row := range residual {
			residual[row].grad += result[row].grad
		}
	})
	return result
}

func softmax(values []float64) []float64 {
	maximum := values[0]
	for _, item := range values[1:] {
		maximum = max(maximum, item)
	}
	result := make([]float64, len(values))
	var sum float64
	for index, item := range values {
		result[index] = math.Exp(item - maximum)
		sum += result[index]
	}
	for index := range result {
		result[index] /= sum
	}
	return result
}

func matrixMultiply(dst, left []float64, leftRows, shared int, right []float64, rightCols int) {
	for row := range leftRows {
		output := dst[row*rightCols : (row+1)*rightCols]
		leftRow := left[row*shared : (row+1)*shared]
		for inner, leftValue := range leftRow {
			if leftValue == 0 {
				continue
			}
			rightRow := right[inner*rightCols : (inner+1)*rightCols]
			for column, rightValue := range rightRow {
				output[column] += leftValue * rightValue
			}
		}
	}
}

func matrixMultiplyBT(dst, left []float64, leftRows, shared int, right []float64, rightRows int) {
	for row := range leftRows {
		for rightRow := range rightRows {
			var sum float64
			for inner := range shared {
				sum += left[row*shared+inner] * right[rightRow*shared+inner]
			}
			dst[row*rightRows+rightRow] = sum
		}
	}
}

func matrixMultiplyTransA(dst, left []float64, leftRows, leftCols int, right []float64, rightCols int) {
	for row := range leftRows {
		leftRow := left[row*leftCols : (row+1)*leftCols]
		rightRow := right[row*rightCols : (row+1)*rightCols]
		for leftColumn, leftValue := range leftRow {
			if leftValue == 0 {
				continue
			}
			output := dst[leftColumn*rightCols : (leftColumn+1)*rightCols]
			for rightColumn, rightValue := range rightRow {
				output[rightColumn] += leftValue * rightValue
			}
		}
	}
}
