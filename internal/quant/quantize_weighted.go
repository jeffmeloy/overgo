package quant

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"

	"llamacpp2go/internal/tensor/dtype"
)

var (
	iq2XXSQuantCodebook = buildIQ2QuantCodebook(iq2XXSGrid[:])
	iq2XSQuantCodebook  = buildIQ2QuantCodebook(iq2XSGrid[:])
	iq1Codebook         = buildIQ1QuantCodebook(iq1SGrid[:])
)

const (
	iq2XXSWeightedGroupWidth = 32
	iq2XXSWeightedTypeSize   = 66
	iq2XXSWeightedAttempts   = 6
	iq2XSWeightedGroupWidth  = 16
	iq2XSWeightedTypeSize    = 74
	iq2XSWeightedAttempts    = 9
	iq2ScaleHeaderBytes      = 2
	iq2XXSGroupBytes         = 8
	iq2XXSSignWordOffset     = 6
	iq2XSGridBytes           = 2
	iq2XSScaleOffset         = 66
)

type iq1QuantCodebook struct {
	lanes [][8]int8
	index map[uint16]int
}

func buildIQ1QuantCodebook(grid []uint64) iq1QuantCodebook {
	result := iq1QuantCodebook{
		lanes: make([][8]int8, len(grid)),
		index: make(map[uint16]int, len(grid)),
	}
	for gridIndex, packed := range grid {
		var encoded uint16
		for lane := 0; lane < 8; lane++ {
			value := int8(byte(packed >> uint(lane*8)))
			result.lanes[gridIndex][lane] = value
			encoded |= uint16(value+1) << uint(2*lane)
		}
		result.index[encoded] = gridIndex
	}
	return result
}

func quantizeIQ2Weighted(dataType dtype.Type, values, importance []float32, output []byte) error {
	codebook := iq2XXSQuantCodebook
	groupWidth := iq2XXSWeightedGroupWidth
	typeSize := iq2XXSWeightedTypeSize
	attempts := iq2XXSWeightedAttempts
	if dataType == dtype.IQ2XS {
		codebook = iq2XSQuantCodebook
		groupWidth = iq2XSWeightedGroupWidth
		typeSize = iq2XSWeightedTypeSize
		attempts = iq2XSWeightedAttempts
	}
	for block := 0; block < len(values)/iqSuperBlockWidth; block++ {
		input := values[block*iqSuperBlockWidth : (block+1)*iqSuperBlockWidth]
		weights := importance[block*iqSuperBlockWidth : (block+1)*iqSuperBlockWidth]
		destination := output[block*typeSize : (block+1)*typeSize]
		for _, value := range input {
			if !finiteFloat32(value) {
				return fmt.Errorf("%s input contains a non-finite value", dataType)
			}
		}
		var sumSquares float32
		for _, value := range input {
			sumSquares += value * value
		}
		variance := sumSquares / iqSuperBlockWidth
		scales := make([]float32, iqSuperBlockWidth/groupWidth)
		var maxScale float32
		for group := range scales {
			start := group * groupWidth
			groupInput := input[start : start+groupWidth]
			groupImportance := weights[start : start+groupWidth]
			weight := make([]float32, groupWidth)
			neighborWeight := make([]float32, groupWidth)
			absoluteValues := make([]float32, groupWidth)
			levels := make([]int8, groupWidth)
			auxiliary := make([]int8, groupWidth)
			signs := make([]byte, groupWidth/iqCodebookLaneWidth)
			for index, value := range groupInput {
				weight[index] = groupImportance[index] * float32(math.Sqrt(float64(variance+value*value)))
				neighborWeight[index] = float32(math.Sqrt(float64(weight[index])))
				absoluteValues[index] = absoluteFloat32(value)
				if value < 0 {
					signs[index/iqCodebookLaneWidth] |= 1 << uint(index%iqCodebookLaneWidth)
				}
			}
			for signGroup := range signs {
				if bitsSet(signs[signGroup])%2 != 0 {
					minimumIndex := signGroup * iqCodebookLaneWidth
					minimum := weight[minimumIndex] * groupInput[minimumIndex] * groupInput[minimumIndex]
					for lane := 1; lane < iqCodebookLaneWidth; lane++ {
						index := signGroup*iqCodebookLaneWidth + lane
						score := weight[index] * groupInput[index] * groupInput[index]
						if score < minimum {
							minimum, minimumIndex = score, index
						}
					}
					absoluteValues[minimumIndex] = -absoluteValues[minimumIndex]
					signs[signGroup] ^= 1 << uint(minimumIndex%iqCodebookLaneWidth)
				}
				signs[signGroup] &= 0x7f
			}
			maximum := absoluteValues[0]
			for _, value := range absoluteValues[1:] {
				maximum = max(maximum, value)
			}
			if maximum < iq2MinimumMagnitude {
				continue
			}
			scale := maximum / 5
			effectiveMaximum := maximum
			if dataType == dtype.IQ2XXS {
				scale = makeQPQuants(absoluteValues, levels, weight, 4)
				effectiveMaximum = scale * 3
				if effectiveMaximum <= 0 {
					continue
				}
			}
			best := float32(0)
			onGrid := make([]bool, groupWidth/iqCodebookLaneWidth)
			auxiliaryOnGrid := make([]bool, groupWidth/iqCodebookLaneWidth)
			for attempt := -attempts; attempt <= attempts; attempt++ {
				inverse := (5 + float32(attempt)*0.1) / effectiveMaximum
				candidateScale := 1 / inverse
				for subGroup := 0; subGroup < groupWidth/iqCodebookLaneWidth; subGroup++ {
					var encoded uint16
					for lane := 0; lane < iqCodebookLaneWidth; lane++ {
						index := subGroup*iqCodebookLaneWidth + lane
						level := nearestIntGGML(0.5 * (inverse*absoluteValues[index] - 1))
						level = max(0, min(2, level))
						auxiliary[index] = int8(level)
						encoded |= uint16(level) << uint(2*lane)
					}
					_, direct := codebook.index[encoded]
					auxiliaryOnGrid[subGroup] = direct
					if !direct {
						iq2FindBest(codebook, encoded,
							absoluteValues[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							neighborWeight[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							candidateScale, auxiliary[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth], 2)
					}
				}
				var sumValue, sumQuantized float32
				for index, level := range auxiliary {
					quantized := float32(2*level + 1)
					sumValue += weight[index] * absoluteValues[index] * quantized
					sumQuantized += weight[index] * quantized * quantized
				}
				if sumQuantized > 0 && sumValue*sumValue > best*sumQuantized {
					scale = sumValue / sumQuantized
					best = scale * sumValue
					copy(levels, auxiliary)
					copy(onGrid, auxiliaryOnGrid)
				}
			}
			if scale > 0 {
				inverse := 1 / scale
				for subGroup := 0; subGroup < groupWidth/iqCodebookLaneWidth; subGroup++ {
					if dataType == dtype.IQ2XS && onGrid[subGroup] {
						continue
					}
					var encoded uint16
					for lane := 0; lane < iqCodebookLaneWidth; lane++ {
						index := subGroup*iqCodebookLaneWidth + lane
						level := nearestIntGGML(0.5 * (inverse*absoluteValues[index] - 1))
						level = max(0, min(2, level))
						if dataType == dtype.IQ2XS {
							levels[index] = int8(level)
						}
						encoded |= uint16(level) << uint(2*lane)
					}
					if _, direct := codebook.index[encoded]; !direct {
						iq2FindBest(codebook, encoded,
							absoluteValues[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							neighborWeight[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							scale, levels[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth], 2)
					}
				}
				var sumValue, sumQuantized float32
				for index, level := range levels {
					quantized := float32(2*level + 1)
					sumValue += weight[index] * absoluteValues[index] * quantized
					sumQuantized += weight[index] * quantized * quantized
				}
				if sumQuantized > 0 {
					scale = sumValue / sumQuantized
				}
			}
			if scale < 0 {
				scale = -scale
				for index := range signs {
					signs[index] = ^signs[index] & 0x7f
				}
			}
			if dataType == dtype.IQ2XXS {
				var first, second uint32
				for subGroup := range signs {
					encoded := encodeIQ2Levels(levels[subGroup*iqCodebookLaneWidth : (subGroup+1)*iqCodebookLaneWidth])
					gridIndex, ok := codebook.index[encoded]
					if !ok {
						return errors.New("IQ2_XXS quantized point is not on the grid")
					}
					first |= uint32(gridIndex) << uint(8*subGroup)
					second |= uint32(signs[subGroup]) << uint(7*subGroup)
				}
				groupOffset := group * iq2XXSGroupBytes
				binary.LittleEndian.PutUint32(destination[iq2ScaleHeaderBytes+groupOffset:], first)
				binary.LittleEndian.PutUint32(destination[iq2XXSSignWordOffset+groupOffset:], second)
			} else {
				for subGroup := range signs {
					encoded := encodeIQ2Levels(levels[subGroup*iqCodebookLaneWidth : (subGroup+1)*iqCodebookLaneWidth])
					gridIndex, ok := codebook.index[encoded]
					if !ok {
						return errors.New("IQ2_XS quantized point is not on the grid")
					}
					packed := uint16(gridIndex) | uint16(signs[subGroup])<<9
					offset := iq2ScaleHeaderBytes + (group*2+subGroup)*iq2XSGridBytes
					binary.LittleEndian.PutUint16(destination[offset:], packed)
				}
			}
			scales[group] = scale
			maxScale = max(maxScale, scale)
		}
		if maxScale == 0 {
			continue
		}
		scale := maxScale / 31
		binary.LittleEndian.PutUint16(destination, Float32ToFloat16(scale))
		inverse := 1 / scale
		for group, groupScale := range scales {
			quantized := nearestIntGGML(0.5 * (inverse*groupScale - 1))
			quantized = max(0, min(15, quantized))
			if dataType == dtype.IQ2XXS {
				offset := iq2XXSSignWordOffset + group*iq2XXSGroupBytes
				packed := binary.LittleEndian.Uint32(destination[offset:])
				binary.LittleEndian.PutUint32(destination[offset:], packed|uint32(quantized)<<28)
			} else if group%2 == 0 {
				destination[iq2XSScaleOffset+group/2] = byte(quantized)
			} else {
				destination[iq2XSScaleOffset+group/2] |= byte(quantized << 4)
			}
		}
	}
	return nil
}

func makeQPQuants(values []float32, levels []int8, weights []float32, maximumLevel int) float32 {
	maximum := float32(0)
	for _, value := range values {
		maximum = max(maximum, value)
	}
	if maximum < negligibleQuantizationMagnitude {
		clear(levels)
		return 0
	}
	inverse := float32(maximumLevel) / maximum
	for index, value := range values {
		levels[index] = int8(nearestIntGGML(inverse * value))
	}
	scale := 1 / inverse
	bestError := float32(0)
	for index, value := range values {
		difference := value - scale*float32(levels[index])
		bestError += weights[index] * difference * difference
	}
	for attempt := -4; attempt <= 4; attempt++ {
		if attempt == 0 {
			continue
		}
		candidateInverse := (float32(maximumLevel) + 0.1*float32(attempt)) / maximum
		candidateScale := 1 / candidateInverse
		var candidateError float32
		for index, value := range values {
			level := min(maximumLevel, nearestIntGGML(candidateInverse*value))
			difference := value - candidateScale*float32(level)
			candidateError += weights[index] * difference * difference
		}
		if candidateError < bestError {
			bestError, inverse = candidateError, candidateInverse
		}
	}
	var sumValue, sumQuantized float32
	for index, value := range values {
		level := min(maximumLevel, nearestIntGGML(inverse*value))
		levels[index] = int8(level)
		sumValue += weights[index] * value * float32(level)
		sumQuantized += weights[index] * float32(level*level)
	}
	for attempt := 0; attempt < 5; attempt++ {
		changed := false
		for index, value := range values {
			old := float32(levels[index])
			candidateValue := sumValue - weights[index]*value*old
			candidateQuantized := sumQuantized - weights[index]*old*old
			if candidateValue <= 0 || candidateQuantized <= 0 {
				continue
			}
			level := min(maximumLevel, nearestIntGGML(value*candidateQuantized/candidateValue))
			if level == int(levels[index]) {
				continue
			}
			candidateValue += weights[index] * value * float32(level)
			candidateQuantized += weights[index] * float32(level*level)
			if candidateValue*candidateValue*sumQuantized > sumValue*sumValue*candidateQuantized {
				levels[index] = int8(level)
				sumValue, sumQuantized, changed = candidateValue, candidateQuantized, true
			}
		}
		if !changed {
			break
		}
	}
	if sumQuantized > 0 {
		return sumValue / sumQuantized
	}
	return 0
}

type iq1SortedValue struct {
	value float32
	index int
}

func sortedIQ1Values(values []float32) []iq1SortedValue {
	result := make([]iq1SortedValue, len(values))
	for index, value := range values {
		result[index] = iq1SortedValue{value: value, index: index}
	}
	sort.Slice(result, func(left, right int) bool {
		if result[left].value == result[right].value {
			return result[left].index < result[right].index
		}
		return result[left].value < result[right].value
	})
	return result
}

func quantizeIQ1Weighted(dataType dtype.Type, values, importance []float32, output []byte) error {
	if dataType == dtype.IQ1S {
		return quantizeIQ1SWeighted(values, importance, output)
	}
	return quantizeIQ1MWeighted(values, importance, output)
}

func iq1FindBest(encoded uint16, values, weights []float32, scale float32, allowed [3]float32, levels []int8) int {
	target := [8]int8{}
	for lane := range target {
		target[lane] = int8((encoded >> uint(2*lane)) & 3)
	}
	distances := make([]int, len(iq1Codebook.lanes))
	unique := make([]int, 0, 32)
	for index, grid := range iq1Codebook.lanes {
		distance := 0
		for lane, value := range grid {
			difference := int(value+1) - int(target[lane])
			distance += difference * difference
		}
		distances[index] = distance
		found := false
		for _, value := range unique {
			found = found || value == distance
		}
		if !found {
			unique = append(unique, distance)
		}
	}
	sort.Ints(unique)
	threshold := unique[min(3, len(unique))-1]
	bestIndex := -1
	bestError := float32(math.MaxFloat32)
	for index, grid := range iq1Codebook.lanes {
		if distances[index] > threshold {
			continue
		}
		var current float32
		for lane, value := range grid {
			difference := scale*allowed[value+1] - values[lane]
			current += weights[lane] * difference * difference
		}
		if current < bestError {
			bestError, bestIndex = current, index
		}
	}
	for lane, value := range iq1Codebook.lanes[bestIndex] {
		levels[lane] = value + 1
	}
	return bestIndex
}

func quantizeIQ1SWeighted(values, importance []float32, output []byte) error {
	const (
		blockWidth = 256
		groupWidth = 32
		typeSize   = 50
		delta      = float32(0.125)
	)
	positive := [3]float32{-1 + delta, delta, 1 + delta}
	negative := [3]float32{-1 - delta, -delta, 1 - delta}
	for block := 0; block < len(values)/blockWidth; block++ {
		input := values[block*blockWidth : (block+1)*blockWidth]
		importanceBlock := importance[block*blockWidth : (block+1)*blockWidth]
		destination := output[block*typeSize : (block+1)*typeSize]
		for _, value := range input {
			if !finiteFloat32(value) {
				return errors.New("IQ1_S input contains a non-finite value")
			}
		}
		var sumSquares float32
		for _, value := range input {
			sumSquares += value * value
		}
		variance := 2 * sumSquares / blockWidth
		scales := make([]float32, blockWidth/groupWidth)
		shifts := make([]int, len(scales))
		var maxScale float32
		for group := range scales {
			start := group * groupWidth
			groupInput := input[start : start+groupWidth]
			weight := make([]float32, groupWidth)
			maximum := float32(0)
			for index, value := range groupInput {
				weight[index] = importanceBlock[start+index] * float32(math.Sqrt(float64(variance+value*value)))
				maximum = max(maximum, absoluteFloat32(value))
			}
			levels := make([]int8, groupWidth)
			if maximum < iq1MinimumMagnitude {
				for index := range levels {
					levels[index] = 1
				}
				shifts[group] = 1
				continue
			}
			sorted := sortedIQ1Values(groupInput)
			sumValue := make([]float32, groupWidth+1)
			sumWeight := make([]float32, groupWidth+1)
			for index, item := range sorted {
				sumValue[index+1] = sumValue[index] + weight[item.index]*groupInput[item.index]
				sumWeight[index+1] = sumWeight[index] + weight[item.index]
			}
			bestScore := float32(-math.MaxFloat32)
			bestFirst, bestSecond, bestShift := -1, -1, 0
			scale := maximum
			for first := 0; first <= groupWidth; first++ {
				for second := first; second <= groupWidth; second++ {
					for _, candidate := range []struct {
						allowed [3]float32
						shift   int
					}{{positive, 1}, {negative, -1}} {
						qx := (sumValue[first]-sumValue[0])*candidate.allowed[0] +
							(sumValue[second]-sumValue[first])*candidate.allowed[1] +
							(sumValue[groupWidth]-sumValue[second])*candidate.allowed[2]
						q2 := (sumWeight[first]-sumWeight[0])*candidate.allowed[0]*candidate.allowed[0] +
							(sumWeight[second]-sumWeight[first])*candidate.allowed[1]*candidate.allowed[1] +
							(sumWeight[groupWidth]-sumWeight[second])*candidate.allowed[2]*candidate.allowed[2]
						if q2 > 0 && qx*qx > bestScore*q2 {
							scale, bestScore = qx/q2, qx*qx/q2
							bestFirst, bestSecond, bestShift = first, second, candidate.shift
						}
					}
				}
			}
			if bestFirst < 0 {
				for index := range levels {
					levels[index] = 1
				}
				shifts[group] = 1
				continue
			}
			for index, item := range sorted {
				switch {
				case index < bestFirst:
					levels[item.index] = 0
				case index < bestSecond:
					levels[item.index] = 1
				default:
					levels[item.index] = 2
				}
			}
			if scale < 0 {
				for index := range levels {
					levels[index] = 2 - levels[index]
				}
				scale, bestShift = -scale, -bestShift
			}
			allowed := positive
			if bestShift < 0 {
				allowed = negative
			}
			indices := [4]int{}
			allOnGrid := true
			for subGroup := 0; subGroup < 4; subGroup++ {
				encoded := encodeIQ2Levels(levels[subGroup*8 : (subGroup+1)*8])
				gridIndex, ok := iq1Codebook.index[encoded]
				if !ok {
					allOnGrid = false
					gridIndex = iq1FindBest(encoded,
						groupInput[subGroup*8:(subGroup+1)*8], weight[subGroup*8:(subGroup+1)*8],
						scale, allowed, levels[subGroup*8:(subGroup+1)*8])
				}
				indices[subGroup] = gridIndex
			}
			if !allOnGrid {
				var qx, q2 float32
				for subGroup, gridIndex := range indices {
					for lane, value := range iq1Codebook.lanes[gridIndex] {
						index := subGroup*8 + lane
						quantized := allowed[value+1]
						qx += weight[index] * quantized * groupInput[index]
						q2 += weight[index] * quantized * quantized
					}
				}
				if qx > 0 && q2 > 0 {
					scale = qx / q2
				}
			}
			var high uint16
			for subGroup, gridIndex := range indices {
				destination[2+group*4+subGroup] = byte(gridIndex)
				high |= uint16(gridIndex>>8) << uint(3*subGroup)
			}
			binary.LittleEndian.PutUint16(destination[34+group*2:], high)
			scales[group], shifts[group] = scale, bestShift
			maxScale = max(maxScale, scale)
		}
		if maxScale == 0 {
			continue
		}
		scale := maxScale / 15
		binary.LittleEndian.PutUint16(destination, Float32ToFloat16(scale*1.125))
		inverse := 1 / scale
		for group, groupScale := range scales {
			quantized := nearestIntGGML(0.5 * (inverse*groupScale - 1))
			quantized = max(0, min(7, quantized))
			if shifts[group] < 0 {
				quantized |= 8
			}
			offset := 34 + group*2
			high := binary.LittleEndian.Uint16(destination[offset:]) | uint16(quantized)<<12
			binary.LittleEndian.PutUint16(destination[offset:], high)
		}
	}
	return nil
}

func quantizeIQ1MWeighted(values, importance []float32, output []byte) error {
	const (
		blockWidth = 256
		groupWidth = 16
		typeSize   = 56
		delta      = float32(0.125)
	)
	positive := [3]float32{-1 + delta, delta, 1 + delta}
	negative := [3]float32{-1 - delta, -delta, 1 - delta}
	shiftMasks := [4]byte{0, 0x80, 0x08, 0x88}
	for block := 0; block < len(values)/blockWidth; block++ {
		input := values[block*blockWidth : (block+1)*blockWidth]
		importanceBlock := importance[block*blockWidth : (block+1)*blockWidth]
		destination := output[block*typeSize : (block+1)*typeSize]
		for _, value := range input {
			if !finiteFloat32(value) {
				return errors.New("IQ1_M input contains a non-finite value")
			}
		}
		var sumSquares float32
		for _, value := range input {
			sumSquares += value * value
		}
		variance := 2 * sumSquares / blockWidth
		scales := make([]float32, blockWidth/groupWidth)
		shifts := make([]int, len(scales))
		indices := make([][2]int, len(scales))
		var maxScale float32
		for group := range scales {
			start := group * groupWidth
			groupInput := input[start : start+groupWidth]
			weight := make([]float32, groupWidth)
			maximum := float32(0)
			for index, value := range groupInput {
				weight[index] = importanceBlock[start+index] * float32(math.Sqrt(float64(variance+value*value)))
				maximum = max(maximum, absoluteFloat32(value))
			}
			levels := make([]int8, groupWidth)
			if maximum < 1e-7 {
				for index := range levels {
					levels[index] = 1
				}
				continue
			}
			sorted := sortedIQ1Values(groupInput)
			bestScore := float32(-math.MaxFloat32)
			bestFirst, bestSecond, bestPattern := -1, -1, -1
			scale := maximum
			for first := 0; first <= groupWidth; first++ {
				for second := first; second <= groupWidth; second++ {
					var qx, q2 [4]float32
					for position, item := range sorted {
						level := 2
						if position < first {
							level = 0
						} else if position < second {
							level = 1
						}
						for pattern := 0; pattern < 4; pattern++ {
							usePositive := pattern < 2
							if item.index >= 8 {
								usePositive = pattern%2 == 0
							}
							allowed := negative
							if usePositive {
								allowed = positive
							}
							quantized := allowed[level]
							qx[pattern] += weight[item.index] * quantized * groupInput[item.index]
							q2[pattern] += weight[item.index] * quantized * quantized
						}
					}
					for pattern := 0; pattern < 4; pattern++ {
						if q2[pattern] > 0 && qx[pattern]*qx[pattern] > bestScore*q2[pattern] {
							scale, bestScore = qx[pattern]/q2[pattern], qx[pattern]*qx[pattern]/q2[pattern]
							bestFirst, bestSecond, bestPattern = first, second, pattern
						}
					}
				}
			}
			if bestFirst < 0 {
				continue
			}
			for position, item := range sorted {
				switch {
				case position < bestFirst:
					levels[item.index] = 0
				case position < bestSecond:
					levels[item.index] = 1
				default:
					levels[item.index] = 2
				}
			}
			if scale < 0 {
				for index := range levels {
					levels[index] = 2 - levels[index]
				}
				scale = -scale
				bestPattern = [4]int{3, 2, 1, 0}[bestPattern]
			}
			allOnGrid := true
			for subGroup := 0; subGroup < 2; subGroup++ {
				allowed := negative
				if (subGroup == 0 && bestPattern < 2) || (subGroup == 1 && bestPattern%2 == 0) {
					allowed = positive
				}
				encoded := encodeIQ2Levels(levels[subGroup*8 : (subGroup+1)*8])
				gridIndex, ok := iq1Codebook.index[encoded]
				if !ok {
					allOnGrid = false
					gridIndex = iq1FindBest(encoded,
						groupInput[subGroup*8:(subGroup+1)*8], weight[subGroup*8:(subGroup+1)*8],
						scale, allowed, levels[subGroup*8:(subGroup+1)*8])
				}
				indices[group][subGroup] = gridIndex
			}
			if !allOnGrid {
				var qx, q2 float32
				for subGroup, gridIndex := range indices[group] {
					allowed := negative
					if (subGroup == 0 && bestPattern < 2) || (subGroup == 1 && bestPattern%2 == 0) {
						allowed = positive
					}
					for lane, value := range iq1Codebook.lanes[gridIndex] {
						index := subGroup*8 + lane
						quantized := allowed[value+1]
						qx += weight[index] * quantized * groupInput[index]
						q2 += weight[index] * quantized * quantized
					}
				}
				if qx > 0 && q2 > 0 {
					scale = qx / q2
				}
			}
			destination[group*2] = byte(indices[group][0])
			destination[group*2+1] = byte(indices[group][1])
			destination[32+group] = byte(indices[group][0]>>8) |
				byte(indices[group][1]>>8)<<4
			scales[group], shifts[group] = scale, bestPattern
			maxScale = max(maxScale, scale)
		}
		if maxScale == 0 {
			continue
		}
		scaleWords := [4]uint16{}
		scale := maxScale / 15
		inverse := 1 / scale
		var totalValue, totalQuantized float32
		for group, groupScale := range scales {
			level := nearestIntGGML(0.5 * (inverse*groupScale - 1))
			level = max(0, min(7, level))
			scaleWords[group/4] |= uint16(level) << uint(3*(group%4))
			destination[32+group] |= shiftMasks[shifts[group]]
			start := group * groupWidth
			for subGroup, gridIndex := range indices[group] {
				allowed := negative
				if (subGroup == 0 && shifts[group] < 2) || (subGroup == 1 && shifts[group]%2 == 0) {
					allowed = positive
				}
				for lane, value := range iq1Codebook.lanes[gridIndex] {
					index := subGroup*8 + lane
					weight := importanceBlock[start+index] * float32(math.Sqrt(float64(variance+input[start+index]*input[start+index])))
					quantized := allowed[value+1] * float32(2*level+1)
					totalValue += weight * quantized * input[start+index]
					totalQuantized += weight * quantized * quantized
				}
			}
		}
		if totalQuantized > 0 {
			scale = totalValue / totalQuantized
		}
		half := Float32ToFloat16(scale * 1.1125)
		scaleWords[0] |= (half & 0x000f) << 12
		scaleWords[1] |= (half & 0x00f0) << 8
		scaleWords[2] |= (half & 0x0f00) << 4
		scaleWords[3] |= half & 0xf000
		for index, word := range scaleWords {
			binary.LittleEndian.PutUint16(destination[48+index*2:], word)
		}
	}
	return nil
}
