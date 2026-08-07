package quant

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/tensor/dtype"
)

const (
	negligibleQuantizationMagnitude = 1e-15
	iq2MinimumMagnitude             = 1e-8
	iq1MinimumMagnitude             = 1e-12
	iqSuperBlockWidth               = 256
	iqCodebookLaneWidth             = 8
	iq3GroupWidth                   = 32
	iq3CodebookWidth                = 4
	iq2SGroupWidth                  = 16
)

// Quantize: deterministic pinned GGML storage conversion.
func Quantize(dataType dtype.Type, values []float32) ([]byte, error) {
	return quantize(dataType, values, nil)
}

// QuantizeWeighted: importance-weighted pinned GGML storage conversion.
func QuantizeWeighted(dataType dtype.Type, values, weights []float32) ([]byte, error) {
	if !RequiresImportance(dataType) {
		return nil, fmt.Errorf("weighted quantization to %s is not implemented", dataType)
	}
	if len(weights) != len(values) {
		return nil, fmt.Errorf("importance count %d differs from element count %d", len(weights), len(values))
	}
	for index, weight := range weights {
		if !finiteFloat32(weight) || weight < 0 {
			return nil, fmt.Errorf("importance weight %d is invalid", index)
		}
	}
	return quantize(dataType, values, weights)
}

func quantize(dataType dtype.Type, values, weights []float32) ([]byte, error) {
	traits, ok := dataType.Traits()
	if !ok {
		return nil, fmt.Errorf("unknown tensor type %d", dataType)
	}
	if uint64(len(values))%traits.BlockSize != 0 {
		return nil, fmt.Errorf(
			"element count %d is not divisible by %s block size %d",
			len(values),
			traits.Name,
			traits.BlockSize,
		)
	}
	blocks := uint64(len(values)) / traits.BlockSize
	if blocks > uint64(math.MaxInt)/traits.TypeSize {
		return nil, errors.New("quantized tensor exceeds addressable memory")
	}
	output := make([]byte, int(blocks*traits.TypeSize))
	switch dataType {
	case dtype.F32:
		for index, value := range values {
			binary.LittleEndian.PutUint32(output[index*4:], math.Float32bits(value))
		}
	case dtype.F16:
		for index, value := range values {
			binary.LittleEndian.PutUint16(output[index*2:], Float32ToFloat16(value))
		}
	case dtype.BF16:
		for index, value := range values {
			binary.LittleEndian.PutUint16(output[index*2:], dtype.Float32ToBF16(value))
		}
	case dtype.Q1_0:
		if err := quantizeQ1_0(values, output); err != nil {
			return nil, err
		}
	case dtype.Q2_0:
		if err := quantizeQ2_0(values, output); err != nil {
			return nil, err
		}
	case dtype.Q4_0, dtype.Q4_1:
		if err := quantizeQ4(dataType, values, output); err != nil {
			return nil, err
		}
	case dtype.Q5_0, dtype.Q5_1:
		if err := quantizeQ5(dataType, values, output); err != nil {
			return nil, err
		}
	case dtype.Q8_0, dtype.Q8_1:
		if err := quantizeQ8(dataType, values, output); err != nil {
			return nil, err
		}
	case dtype.Q8K:
		if err := quantizeQ8K(values, output); err != nil {
			return nil, err
		}
	case dtype.Q6K:
		if err := quantizeQ6K(values, output); err != nil {
			return nil, err
		}
	case dtype.Q3K:
		if err := quantizeQ3K(values, output); err != nil {
			return nil, err
		}
	case dtype.Q2K:
		if err := quantizeQ2K(values, output); err != nil {
			return nil, err
		}
	case dtype.Q4K, dtype.Q5K:
		if err := quantizeQ4Or5K(dataType, values, output); err != nil {
			return nil, err
		}
	case dtype.TQ1_0, dtype.TQ2_0:
		if err := quantizeTernary(dataType, values, output); err != nil {
			return nil, err
		}
	case dtype.MXFP4:
		if err := quantizeMXFP4(values, output); err != nil {
			return nil, err
		}
	case dtype.NVFP4:
		if err := quantizeNVFP4(values, output); err != nil {
			return nil, err
		}
	case dtype.IQ4NL, dtype.IQ4XS:
		if err := quantizeIQ4(dataType, values, output); err != nil {
			return nil, err
		}
	case dtype.IQ2S:
		if err := quantizeIQ2S(values, output); err != nil {
			return nil, err
		}
	case dtype.IQ2XXS, dtype.IQ2XS:
		if weights == nil {
			return nil, fmt.Errorf("quantization to %s requires importance weights", dataType)
		}
		if err := quantizeIQ2Weighted(dataType, values, weights, output); err != nil {
			return nil, err
		}
	case dtype.IQ1S, dtype.IQ1M:
		if weights == nil {
			return nil, fmt.Errorf("quantization to %s requires importance weights", dataType)
		}
		if err := quantizeIQ1Weighted(dataType, values, weights, output); err != nil {
			return nil, err
		}
	case dtype.IQ3XXS, dtype.IQ3S:
		if err := quantizeIQ3(dataType, values, output); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("quantization to %s is not implemented", dataType)
	}
	return output, nil
}

type iq3QuantCodebook struct {
	lanes [][4]int8
	index map[uint16]int
}

var (
	iq3XXSQuantCodebook = buildIQ3QuantCodebook(iq3XXSGrid[:])
	iq3SQuantCodebook   = buildIQ3QuantCodebook(iq3SGrid[:])
)

func buildIQ3QuantCodebook(grid []uint32) iq3QuantCodebook {
	unique := make([]byte, 0, 8)
	for _, packed := range grid {
		for lane := 0; lane < 4; lane++ {
			value := byte(packed >> uint(lane*8))
			if !slices.Contains(unique, value) {
				unique = append(unique, value)
			}
		}
	}
	slices.Sort(unique)
	result := iq3QuantCodebook{
		lanes: make([][4]int8, len(grid)),
		index: make(map[uint16]int, len(grid)),
	}
	for gridIndex, packed := range grid {
		var encoded uint16
		for lane := 0; lane < 4; lane++ {
			value := byte(packed >> uint(lane*8))
			level, _ := slices.BinarySearch(unique, value)
			result.lanes[gridIndex][lane] = int8(2*level + 1)
			encoded |= uint16(level) << uint(3*lane)
		}
		result.index[encoded] = gridIndex
	}
	return result
}

func quantizeIQ3(
	dataType dtype.Type,
	values []float32,
	output []byte,
) error {
	const blockWidth = iqSuperBlockWidth
	typeSize := iq3XXSBlockBytes
	codebook := iq3XXSQuantCodebook
	distanceTiers := 2
	attempts := 15
	attemptStep := float32(0.2)
	paritySigns := true
	refineAll := false
	scaleFudge := float32(1.0125)
	if dataType == dtype.IQ3S {
		typeSize = iq3SBlockBytes
		codebook = iq3SQuantCodebook
		distanceTiers = 3
		attempts = 9
		paritySigns = false
		refineAll = true
		scaleFudge = 1.033
	}
	for block := 0; block < len(values)/blockWidth; block++ {
		input := values[block*blockWidth : (block+1)*blockWidth]
		destination := output[block*typeSize : (block+1)*typeSize]
		for _, value := range input {
			if !finiteFloat32(value) {
				return fmt.Errorf("%s input contains a non-finite value", dataType)
			}
		}
		scales := make([]float32, blockWidth/iq3GroupWidth)
		var maxScale float32
		for group := 0; group < blockWidth/iq3GroupWidth; group++ {
			indices, signs, scale, err := quantizeIQ3Group(
				input[group*iq3GroupWidth:(group+1)*iq3GroupWidth],
				codebook,
				distanceTiers,
				attempts,
				attemptStep,
				paritySigns,
				refineAll,
			)
			if err != nil {
				return fmt.Errorf("%s: %w", dataType, err)
			}
			if dataType == dtype.IQ3XXS {
				for index, gridIndex := range indices {
					destination[iqPackedGridStart+group*8+index] = byte(gridIndex)
				}
				packedSigns := uint32(signs[0]) |
					uint32(signs[1])<<7 |
					uint32(signs[2])<<14 |
					uint32(signs[3])<<21
				binary.LittleEndian.PutUint32(
					destination[iq3XXSScaleSignStart+group*4:],
					packedSigns,
				)
			} else {
				for index, gridIndex := range indices {
					destination[iqPackedGridStart+group*8+index] = byte(gridIndex)
					destination[iq3SHighStart+group] |=
						byte(gridIndex>>8) << uint(index)
				}
				copy(destination[iq3SSignStart+group*4:], signs[:])
			}
			scales[group] = scale
			maxScale = max(maxScale, scale)
		}
		if maxScale == 0 {
			continue
		}
		scale := maxScale / 31
		binary.LittleEndian.PutUint16(
			destination,
			Float32ToFloat16(scale*scaleFudge),
		)
		inverse := 1 / scale
		for group, groupScale := range scales {
			quantized := nearestIntGGML(0.5 * (inverse*groupScale - 1))
			quantized = max(0, min(15, quantized))
			if dataType == dtype.IQ3XXS {
				offset := iq3XXSScaleSignStart + group*4
				packed := binary.LittleEndian.Uint32(destination[offset:])
				packed |= uint32(quantized) << 28
				binary.LittleEndian.PutUint32(destination[offset:], packed)
			} else if group%2 == 0 {
				destination[iq3SScaleStart+group/2] = byte(quantized)
			} else {
				destination[iq3SScaleStart+group/2] |= byte(quantized << 4)
			}
		}
	}
	return nil
}

func quantizeIQ3Group(
	input []float32,
	codebook iq3QuantCodebook,
	distanceTiers, attempts int,
	attemptStep float32,
	paritySigns, refineAll bool,
) ([iq3GroupWidth / iq3CodebookWidth]int, [iq3GroupWidth / iqCodebookLaneWidth]byte, float32, error) {
	var indices [iq3GroupWidth / iq3CodebookWidth]int
	var signs [iq3GroupWidth / iqCodebookLaneWidth]byte
	weight := make([]float32, iq3GroupWidth)
	neighborWeight := make([]float32, iq3GroupWidth)
	absoluteValues := make([]float32, iq3GroupWidth)
	levels := make([]int8, iq3GroupWidth)
	auxiliary := make([]int8, iq3GroupWidth)
	onGrid := [iq3GroupWidth / iq3CodebookWidth]bool{}
	auxiliaryOnGrid := [iq3GroupWidth / iq3CodebookWidth]bool{}
	for index, value := range input {
		weight[index] = value * value
		neighborWeight[index] = absoluteFloat32(value)
		if value >= 0 {
			absoluteValues[index] = value
		} else {
			absoluteValues[index] = -value
			signs[index/iqCodebookLaneWidth] |= 1 << uint(index%iqCodebookLaneWidth)
		}
	}
	if paritySigns {
		for signGroup := range signs {
			if bitsSet(signs[signGroup])%2 != 0 {
				minimumIndex := 0
				minimum := weight[signGroup*iqCodebookLaneWidth] *
					input[signGroup*iqCodebookLaneWidth] * input[signGroup*iqCodebookLaneWidth]
				for lane := 1; lane < iqCodebookLaneWidth; lane++ {
					index := signGroup*iqCodebookLaneWidth + lane
					score := weight[index] * input[index] * input[index]
					if score < minimum {
						minimum = score
						minimumIndex = lane
					}
				}
				index := signGroup*iqCodebookLaneWidth + minimumIndex
				absoluteValues[index] = -absoluteValues[index]
				signs[signGroup] ^= 1 << uint(minimumIndex)
			}
			signs[signGroup] &= 0x7f
		}
	}
	maximum := absoluteValues[0]
	for _, value := range absoluteValues[1:] {
		maximum = max(maximum, value)
	}
	epsilon := float32(0)
	if paritySigns {
		epsilon = iq2MinimumMagnitude
	}
	if maximum <= epsilon {
		return indices, signs, 0, nil
	}
	best := float32(0)
	scale := maximum / 15
	for attempt := -attempts; attempt <= attempts; attempt++ {
		inverse := (15 + float32(attempt)*attemptStep) / maximum
		candidateScale := 1 / inverse
		for subGroup := range indices {
			var encoded uint16
			for lane := 0; lane < iq3CodebookWidth; lane++ {
				index := subGroup*iq3CodebookWidth + lane
				level := nearestIntGGML(0.5 *
					(inverse*absoluteValues[index] - 1))
				level = max(0, min(7, level))
				auxiliary[index] = int8(level)
				encoded |= uint16(level) << uint(3*lane)
			}
			_, direct := codebook.index[encoded]
			auxiliaryOnGrid[subGroup] = direct
			if !direct {
				iq3FindBest(
					codebook,
					encoded,
					absoluteValues[subGroup*iq3CodebookWidth:(subGroup+1)*iq3CodebookWidth],
					neighborWeight[subGroup*iq3CodebookWidth:(subGroup+1)*iq3CodebookWidth],
					candidateScale,
					auxiliary[subGroup*iq3CodebookWidth:(subGroup+1)*iq3CodebookWidth],
					distanceTiers,
				)
			}
		}
		var sumValue, sumQuantized float32
		for index := 0; index < iq3GroupWidth; index++ {
			quantized := float32(2*auxiliary[index] + 1)
			sumValue += weight[index] * absoluteValues[index] * quantized
			sumQuantized += weight[index] * quantized * quantized
		}
		if sumQuantized > 0 && sumValue*sumValue > best*sumQuantized {
			scale = sumValue / sumQuantized
			best = scale * sumValue
			copy(levels, auxiliary)
			onGrid = auxiliaryOnGrid
		}
	}
	hasOffGrid := slices.Contains(onGrid[:], false)
	if hasOffGrid && scale > 0 {
		inverse := 1 / scale
		for subGroup := range indices {
			if !refineAll && onGrid[subGroup] {
				continue
			}
			var encoded uint16
			for lane := 0; lane < iq3CodebookWidth; lane++ {
				index := subGroup*iq3CodebookWidth + lane
				level := nearestIntGGML(0.5 *
					(inverse*absoluteValues[index] - 1))
				level = max(0, min(7, level))
				encoded |= uint16(level) << uint(3*lane)
			}
			if gridIndex, direct := codebook.index[encoded]; direct {
				for lane, value := range codebook.lanes[gridIndex] {
					levels[subGroup*iq3CodebookWidth+lane] = (value - 1) / 2
				}
			} else {
				iq3FindBest(
					codebook,
					encoded,
					absoluteValues[subGroup*iq3CodebookWidth:(subGroup+1)*iq3CodebookWidth],
					neighborWeight[subGroup*iq3CodebookWidth:(subGroup+1)*iq3CodebookWidth],
					scale,
					levels[subGroup*iq3CodebookWidth:(subGroup+1)*iq3CodebookWidth],
					distanceTiers,
				)
			}
		}
		var sumValue, sumQuantized float32
		for index := 0; index < iq3GroupWidth; index++ {
			quantized := float32(2*levels[index] + 1)
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
			signs[index] = ^signs[index]
			if paritySigns {
				signs[index] &= 0x7f
			}
		}
	}
	for subGroup := range indices {
		var encoded uint16
		for lane := 0; lane < iq3CodebookWidth; lane++ {
			encoded |= uint16(levels[subGroup*iq3CodebookWidth+lane]) << uint(3*lane)
		}
		gridIndex, ok := codebook.index[encoded]
		if !ok {
			return indices, signs, 0,
				errors.New("quantized point is not on the IQ3 grid")
		}
		indices[subGroup] = gridIndex
	}
	return indices, signs, scale, nil
}

func bitsSet(value byte) int {
	count := 0
	for value != 0 {
		value &= value - 1
		count++
	}
	return count
}

func iq3FindBest(
	codebook iq3QuantCodebook,
	encoded uint16,
	values, weights []float32,
	scale float32,
	levels []int8,
	distanceTiers int,
) int {
	target := [4]int8{}
	for lane := 0; lane < 4; lane++ {
		target[lane] = 2*int8((encoded>>uint(3*lane))&7) + 1
	}
	distances := make([]int, len(codebook.lanes))
	uniqueDistances := make([]int, 0, len(codebook.lanes))
	for index, grid := range codebook.lanes {
		distance := 0
		for lane := 0; lane < 4; lane++ {
			difference := int(grid[lane] - target[lane])
			distance += difference * difference
		}
		distances[index] = distance
		if !slices.Contains(uniqueDistances, distance) {
			uniqueDistances = append(uniqueDistances, distance)
		}
	}
	slices.Sort(uniqueDistances)
	threshold := uniqueDistances[min(distanceTiers, len(uniqueDistances))-1]
	bestError := float32(math.MaxFloat32)
	bestIndex := -1
	for index, grid := range codebook.lanes {
		if distances[index] > threshold {
			continue
		}
		currentError := float32(0)
		for lane := 0; lane < 4; lane++ {
			difference := scale*float32(grid[lane]) - values[lane]
			currentError += weights[lane] * difference * difference
		}
		if currentError < bestError {
			bestError = currentError
			bestIndex = index
		}
	}
	for lane, value := range codebook.lanes[bestIndex] {
		levels[lane] = (value - 1) / 2
	}
	return bestIndex
}

type iq2QuantCodebook struct {
	lanes [][8]int8
	index map[uint16]int
}

var iq2SQuantCodebook = buildIQ2QuantCodebook(iq2SGrid[:])

func buildIQ2QuantCodebook(grid []uint64) iq2QuantCodebook {
	unique := make([]byte, 0, 4)
	for _, packed := range grid {
		for lane := 0; lane < 8; lane++ {
			value := byte(packed >> uint(lane*8))
			if !slices.Contains(unique, value) {
				unique = append(unique, value)
			}
		}
	}
	slices.Sort(unique)
	result := iq2QuantCodebook{
		lanes: make([][8]int8, len(grid)),
		index: make(map[uint16]int, len(grid)),
	}
	for gridIndex, packed := range grid {
		var encoded uint16
		for lane := 0; lane < 8; lane++ {
			value := byte(packed >> uint(lane*8))
			level, _ := slices.BinarySearch(unique, value)
			result.lanes[gridIndex][lane] = int8(2*level + 1)
			encoded |= uint16(level) << uint(2*lane)
		}
		result.index[encoded] = gridIndex
	}
	return result
}

func quantizeIQ2S(values []float32, output []byte) error {
	const (
		blockWidth       = iqSuperBlockWidth
		groupWidth       = iq2SGroupWidth
		subGroupCount    = groupWidth / iqCodebookLaneWidth
		typeSize         = iq2SBlockBytes
		gridLowOffset    = iq2SGridStart
		signOffset       = iq2SSignStart
		gridHighOffset   = iq2SHighStart
		groupScaleOffset = iq2SScaleStart
	)
	for block := 0; block < len(values)/blockWidth; block++ {
		input := values[block*blockWidth : (block+1)*blockWidth]
		destination := output[block*typeSize : (block+1)*typeSize]
		sumSquares := float32(0)
		for _, value := range input {
			if !finiteFloat32(value) {
				return errors.New("IQ2_S input contains a non-finite value")
			}
			sumSquares += value * value
		}
		variance := 2 * sumSquares / blockWidth
		scales := make([]float32, blockWidth/groupWidth)
		var maxScale float32
		for group := 0; group < blockWidth/groupWidth; group++ {
			groupInput := input[group*groupWidth : (group+1)*groupWidth]
			weight := make([]float32, groupWidth)
			neighborWeight := make([]float32, groupWidth)
			absoluteValues := make([]float32, groupWidth)
			levels := make([]int8, groupWidth)
			auxiliary := make([]int8, groupWidth)
			onGrid := [subGroupCount]bool{true, true}
			auxiliaryOnGrid := [subGroupCount]bool{}
			signs := [subGroupCount]byte{}
			maximum := float32(0)
			for index, value := range groupInput {
				weight[index] = 0.25*variance + value*value
				neighborWeight[index] = float32(
					math.Sqrt(float64(weight[index])),
				)
				if value >= 0 {
					absoluteValues[index] = value
				} else {
					absoluteValues[index] = -value
					signs[index/8] |= 1 << uint(index%8)
				}
				maximum = max(maximum, absoluteValues[index])
			}
			if maximum < iq2MinimumMagnitude {
				continue
			}
			best := float32(0)
			scale := maximum / 5
			for attempt := -9; attempt <= 9; attempt++ {
				inverse := (5 + float32(attempt)*0.1) / maximum
				candidateScale := 1 / inverse
				for subGroup := range signs {
					var encoded uint16
					for lane := 0; lane < iqCodebookLaneWidth; lane++ {
						level := nearestIntGGML(0.5 *
							(inverse*absoluteValues[subGroup*iqCodebookLaneWidth+lane] - 1))
						level = max(0, min(2, level))
						auxiliary[subGroup*iqCodebookLaneWidth+lane] = int8(level)
						encoded |= uint16(level) << uint(2*lane)
					}
					_, direct := iq2SQuantCodebook.index[encoded]
					auxiliaryOnGrid[subGroup] = direct
					if !direct {
						iq2FindBest(
							iq2SQuantCodebook,
							encoded,
							absoluteValues[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							neighborWeight[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							candidateScale,
							auxiliary[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							1,
						)
					}
				}
				var sumValue, sumQuantized float32
				for index := 0; index < groupWidth; index++ {
					quantized := float32(2*auxiliary[index] + 1)
					sumValue += weight[index] *
						absoluteValues[index] * quantized
					sumQuantized += weight[index] *
						quantized * quantized
				}
				if sumQuantized > 0 &&
					sumValue*sumValue > best*sumQuantized {
					scale = sumValue / sumQuantized
					best = scale * sumValue
					copy(levels, auxiliary)
					onGrid = auxiliaryOnGrid
				}
			}
			if (!onGrid[0] || !onGrid[1]) && scale > 0 {
				inverse := 1 / scale
				for subGroup := range signs {
					if onGrid[subGroup] {
						continue
					}
					var encoded uint16
					for lane := 0; lane < iqCodebookLaneWidth; lane++ {
						level := nearestIntGGML(0.5 *
							(inverse*absoluteValues[subGroup*iqCodebookLaneWidth+lane] - 1))
						level = max(0, min(2, level))
						levels[subGroup*iqCodebookLaneWidth+lane] = int8(level)
						encoded |= uint16(level) << uint(2*lane)
					}
					if _, direct := iq2SQuantCodebook.index[encoded]; !direct {
						iq2FindBest(
							iq2SQuantCodebook,
							encoded,
							absoluteValues[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							neighborWeight[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							scale,
							levels[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							1,
						)
					}
				}
				var sumValue, sumQuantized float32
				for index := 0; index < groupWidth; index++ {
					quantized := float32(2*levels[index] + 1)
					sumValue += weight[index] *
						absoluteValues[index] * quantized
					sumQuantized += weight[index] *
						quantized * quantized
				}
				if sumQuantized > 0 {
					scale = sumValue / sumQuantized
				}
			}
			if scale < 0 {
				scale = -scale
				signs[0] = ^signs[0]
				signs[1] = ^signs[1]
			}
			for subGroup := range signs {
				encoded := encodeIQ2Levels(levels[subGroup*iqCodebookLaneWidth : (subGroup+1)*iqCodebookLaneWidth])
				gridIndex, ok := iq2SQuantCodebook.index[encoded]
				if !ok {
					return errors.New("IQ2_S quantized point is not on the grid")
				}
				index := 2*group + subGroup
				destination[gridLowOffset+index] = byte(gridIndex)
				destination[gridHighOffset+index/4] |=
					byte(gridIndex>>8) << uint(2*(index%4))
				destination[signOffset+index] = signs[subGroup]
			}
			scales[group] = scale
			maxScale = max(maxScale, scale)
		}
		if maxScale == 0 {
			continue
		}
		scale := maxScale / 31
		binary.LittleEndian.PutUint16(
			destination,
			Float32ToFloat16(scale*0.9875),
		)
		inverse := 1 / scale
		for group, groupScale := range scales {
			quantized := nearestIntGGML(0.5 * (inverse*groupScale - 1))
			quantized = max(0, min(15, quantized))
			if group%2 == 0 {
				destination[groupScaleOffset+group/2] = byte(quantized)
			} else {
				destination[groupScaleOffset+group/2] |= byte(quantized << 4)
			}
		}
	}
	return nil
}

func encodeIQ2Levels(levels []int8) uint16 {
	var encoded uint16
	for lane, level := range levels {
		encoded |= uint16(level) << uint(2*lane)
	}
	return encoded
}

func iq2FindBest(
	codebook iq2QuantCodebook,
	encoded uint16,
	values, weights []float32,
	scale float32,
	levels []int8,
	distanceTiers int,
) int {
	target := [8]int8{}
	for lane := 0; lane < 8; lane++ {
		target[lane] = 2*int8((encoded>>uint(2*lane))&3) + 1
	}
	distances := make([]int, len(codebook.lanes))
	uniqueDistances := make([]int, 0, len(codebook.lanes))
	for index, grid := range codebook.lanes {
		distance := 0
		for lane := 0; lane < 8; lane++ {
			difference := int(grid[lane] - target[lane])
			distance += difference * difference
		}
		distances[index] = distance
		if !slices.Contains(uniqueDistances, distance) {
			uniqueDistances = append(uniqueDistances, distance)
		}
	}
	slices.Sort(uniqueDistances)
	threshold := uniqueDistances[min(distanceTiers, len(uniqueDistances))-1]
	bestError := float32(math.MaxFloat32)
	bestIndex := -1
	for index, grid := range codebook.lanes {
		if distances[index] > threshold {
			continue
		}
		currentError := float32(0)
		for lane := 0; lane < 8; lane++ {
			difference := scale*float32(grid[lane]) - values[lane]
			currentError += weights[lane] * difference * difference
		}
		if currentError < bestError {
			bestError = currentError
			bestIndex = index
		}
	}
	for lane, value := range codebook.lanes[bestIndex] {
		levels[lane] = (value - 1) / 2
	}
	return bestIndex
}

func quantizeIQ4(
	dataType dtype.Type,
	values []float32,
	output []byte,
) error {
	const groupWidth = 32
	superBlockWidth := 32
	typeSize := 18
	quantizedOffset := 2
	attempts := -1
	if dataType == dtype.IQ4XS {
		superBlockWidth = 256
		typeSize = 136
		quantizedOffset = 8
		attempts = 7
	}
	for block := 0; block < len(values)/superBlockWidth; block++ {
		input := values[block*superBlockWidth : (block+1)*superBlockWidth]
		destination := output[block*typeSize : (block+1)*typeSize]
		levels := make([]byte, superBlockWidth)
		scales := make([]float32, superBlockWidth/groupWidth)
		var maxScale, maxAbsoluteScale float32
		for group := 0; group < superBlockWidth/groupWidth; group++ {
			scale, err := quantizeIQ4Group(
				input[group*groupWidth:(group+1)*groupWidth],
				levels[group*groupWidth:(group+1)*groupWidth],
				attempts,
			)
			if err != nil {
				return fmt.Errorf("%s: %w", dataType, err)
			}
			scales[group] = scale
			absolute := absoluteFloat32(scale)
			if absolute > maxAbsoluteScale {
				maxAbsoluteScale = absolute
				maxScale = scale
			}
		}
		if dataType == dtype.IQ4XS {
			scale := -maxScale / 32
			binary.LittleEndian.PutUint16(
				destination,
				Float32ToFloat16(scale),
			)
			inverse := float32(0)
			if scale != 0 {
				inverse = 1 / scale
			}
			highScales := uint16(0)
			for group := 0; group < superBlockWidth/groupWidth; group++ {
				quantizedScale := nearestIntGGML(inverse * scales[group])
				quantizedScale = max(-32, min(31, quantizedScale))
				groupScale := scale * float32(quantizedScale)
				groupInverse := float32(0)
				if groupScale != 0 {
					groupInverse = 1 / groupScale
				}
				for index := 0; index < groupWidth; index++ {
					levels[group*groupWidth+index] = byte(bestIQ4Index(
						groupInverse * input[group*groupWidth+index],
					))
				}
				encodedScale := quantizedScale + 32
				if group%2 == 0 {
					destination[4+group/2] = byte(encodedScale & 0x0f)
				} else {
					destination[4+group/2] |= byte(
						(encodedScale & 0x0f) << 4,
					)
				}
				highScales |= uint16(encodedScale>>4) << uint(2*group)
			}
			binary.LittleEndian.PutUint16(destination[2:], highScales)
		} else {
			binary.LittleEndian.PutUint16(
				destination,
				Float32ToFloat16(scales[0]),
			)
		}
		for group := 0; group < superBlockWidth/groupWidth; group++ {
			for lane := 0; lane < groupWidth/2; lane++ {
				destination[quantizedOffset+group*16+lane] =
					levels[group*groupWidth+lane] |
						levels[group*groupWidth+lane+16]<<4
			}
		}
	}
	return nil
}

func quantizeIQ4Group(
	input []float32,
	levels []byte,
	attempts int,
) (float32, error) {
	var maximum, maximumAbsolute float32
	for _, value := range input {
		if !finiteFloat32(value) {
			return 0, errors.New("input contains a non-finite value")
		}
		absolute := absoluteFloat32(value)
		if absolute > maximumAbsolute {
			maximumAbsolute = absolute
			maximum = value
		}
	}
	if maximumAbsolute < negligibleQuantizationMagnitude {
		clear(levels)
		return 0, nil
	}
	scale := maximum / iq4NLValues[0]
	if attempts > 0 {
		scale = -maximum / iq4NLValues[0]
	}
	inverse := 1 / scale
	var sumValue, sumSquared float32
	for index, value := range input {
		level := bestIQ4Index(inverse * value)
		levels[index] = byte(level)
		quantized := iq4NLValues[level]
		weight := value * value
		sumValue += weight * quantized * value
		sumSquared += weight * quantized * quantized
	}
	if sumSquared > 0 {
		scale = sumValue / sumSquared
	} else {
		scale = 0
	}
	best := scale * sumValue
	for attempt := -attempts; attempt <= attempts; attempt++ {
		inverse = (float32(attempt) + iq4NLValues[0]) / maximum
		sumValue = 0
		sumSquared = 0
		for _, value := range input {
			level := bestIQ4Index(inverse * value)
			quantized := iq4NLValues[level]
			weight := value * value
			sumValue += weight * quantized * value
			sumSquared += weight * quantized * quantized
		}
		if sumSquared > 0 && sumValue*sumValue > best*sumSquared {
			scale = sumValue / sumSquared
			best = scale * sumValue
		}
	}
	return scale, nil
}

func bestIQ4Index(value float32) int {
	if value <= iq4NLValues[0] {
		return 0
	}
	if value >= iq4NLValues[len(iq4NLValues)-1] {
		return len(iq4NLValues) - 1
	}
	lower := 0
	upper := len(iq4NLValues) - 1
	for upper-lower > 1 {
		middle := (lower + upper) / 2
		if value < iq4NLValues[middle] {
			upper = middle
		} else {
			lower = middle
		}
	}
	if value-iq4NLValues[upper-1] <
		iq4NLValues[upper]-value {
		return upper - 1
	}
	return upper
}

func quantizeNVFP4(values []float32, output []byte) error {
	const (
		width         = 64
		subBlockWidth = 16
		typeSize      = 36
	)
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*typeSize : (block+1)*typeSize]
		for subBlock := 0; subBlock < width/subBlockWidth; subBlock++ {
			subInput := input[subBlock*subBlockWidth : (subBlock+1)*subBlockWidth]
			maximum, err := maximumAbsolute(subInput, "NVFP4")
			if err != nil {
				return err
			}
			encodedScale := float32ToUE4M3(maximum / 6)
			destination[subBlock] = encodedScale
			scale := ue4m3ToFloat32(encodedScale)
			for lane := 0; lane < subBlockWidth/2; lane++ {
				low := nearestMXFP4(subInput[lane], scale)
				high := nearestMXFP4(subInput[lane+subBlockWidth/2], scale)
				destination[4+subBlock*8+lane] = byte(low | high<<4)
			}
		}
	}
	return nil
}

func float32ToUE4M3(value float32) byte {
	if !(value > 0) {
		return 0
	}
	if value > 448 {
		value = 448
	}
	bits := math.Float32bits(value)
	exponent := int((bits>>23)&0xff) - 127
	mantissa := int((bits >> 20) & 0x07)
	encodedExponent := exponent + 7
	if encodedExponent <= 0 {
		encodedMantissa := int(value*512 + 0.5)
		encodedMantissa = min(7, encodedMantissa)
		if encodedMantissa < 1 {
			return 0
		}
		return byte(encodedMantissa)
	}
	if encodedExponent >= 15 {
		return 0x7e
	}
	mantissa += int((bits >> 19) & 1)
	if mantissa > 7 {
		mantissa = 0
		encodedExponent++
		if encodedExponent >= 15 {
			return 0x7e
		}
	}
	return byte(encodedExponent<<3 | mantissa)
}

func quantizeTernary(
	dataType dtype.Type,
	values []float32,
	output []byte,
) error {
	const width = tqBlockWidth
	typeSize := tq1BlockBytes
	if dataType == dtype.TQ2_0 {
		typeSize = tq2BlockBytes
	}
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*typeSize : (block+1)*typeSize]
		maximum, err := maximumAbsolute(input, dataType.String())
		if err != nil {
			return err
		}
		inverse := float32(0)
		if maximum != 0 {
			inverse = 1 / maximum
		}
		scaleOffset := tq1ScaleStart
		if dataType == dtype.TQ2_0 {
			scaleOffset = tq2ScaleStart
		}
		binary.LittleEndian.PutUint16(
			destination[scaleOffset:],
			Float32ToFloat16(maximum),
		)
		if dataType == dtype.TQ2_0 {
			for section := 0; section < width; section += 128 {
				for lane := 0; lane < 32; lane++ {
					quantized := byte(0)
					for group := 0; group < 4; group++ {
						level := int(roundFloat32(
							input[section+lane+group*32]*inverse,
						)) + 1
						quantized |= byte(level&3) << uint(2*group)
					}
					destination[section/4+lane] = quantized
				}
			}
			continue
		}
		for lane := 0; lane < 32; lane++ {
			quantized := byte(0)
			for group := 0; group < 5; group++ {
				level := int(roundFloat32(
					input[lane+group*32]*inverse,
				)) + 1
				quantized = quantized*3 + byte(level)
			}
			destination[lane] = byte(
				(uint16(quantized)*256 + 242) / 243,
			)
		}
		for lane := 0; lane < 16; lane++ {
			quantized := byte(0)
			for group := 0; group < 5; group++ {
				level := int(roundFloat32(
					input[160+lane+group*16]*inverse,
				)) + 1
				quantized = quantized*3 + byte(level)
			}
			destination[32+lane] = byte(
				(uint16(quantized)*256 + 242) / 243,
			)
		}
		for lane := 0; lane < 4; lane++ {
			quantized := byte(0)
			for group := 0; group < 4; group++ {
				level := int(roundFloat32(
					input[240+lane+group*4]*inverse,
				)) + 1
				quantized = quantized*3 + byte(level)
			}
			quantized *= 3
			destination[48+lane] = byte(
				(uint16(quantized)*256 + 242) / 243,
			)
		}
	}
	return nil
}

func quantizeQ4Or5K(
	dataType dtype.Type,
	values []float32,
	output []byte,
) error {
	layout := q4KCodec
	maxLevel := 15
	if dataType == dtype.Q5K {
		layout = q5KCodec
		maxLevel = 31
	}
	for block := 0; block < len(values)/layout.block.width; block++ {
		input := layout.block.input(values, block)
		destination := layout.block.storage(output, block)
		levels := make([]byte, layout.block.width)
		auxiliary := make([]byte, 32)
		weights := make([]float32, 32)
		minima := make([]float32, layout.block.width/32)
		scales := make([]float32, layout.block.width/32)
		var maxScale, maxMinimum float32
		for group := 0; group < layout.block.width/32; group++ {
			groupInput := input[group*32 : (group+1)*32]
			sumSquares := float32(0)
			for _, value := range groupInput {
				if !finiteFloat32(value) {
					return fmt.Errorf(
						"%s input contains a non-finite value",
						dataType,
					)
				}
				sumSquares += value * value
			}
			average := float32(math.Sqrt(float64(sumSquares / 32)))
			for index, value := range groupInput {
				weights[index] = average + absoluteFloat32(value)
			}
			rangeMinimum := float32(-1)
			steps := 20
			if dataType == dtype.Q5K {
				rangeMinimum = -0.5
				steps = 15
			}
			scale, minimum, err := makeQKX2Quants(
				groupInput,
				maxLevel,
				weights,
				levels[group*32:(group+1)*32],
				auxiliary,
				rangeMinimum,
				0.1,
				steps,
				false,
			)
			if err != nil {
				return fmt.Errorf("%s: %w", dataType, err)
			}
			scales[group] = scale
			minima[group] = minimum
			if scale > maxScale {
				maxScale = scale
			}
			if minimum > maxMinimum {
				maxMinimum = minimum
			}
		}
		inverseScale := float32(0)
		if maxScale > 0 {
			inverseScale = 63 / maxScale
		}
		inverseMinimum := float32(0)
		if maxMinimum > 0 {
			inverseMinimum = 63 / maxMinimum
		}
		for group := 0; group < layout.block.width/32; group++ {
			scale := min(63, nearestIntGGML(inverseScale*scales[group]))
			minimum := min(
				63,
				nearestIntGGML(inverseMinimum*minima[group]),
			)
			setKScaleMinimum(
				layout.scales.bytes(destination),
				group,
				scale,
				minimum,
			)
		}
		scaleBits := Float32ToFloat16(maxScale / 63)
		minimumBits := Float32ToFloat16(maxMinimum / 63)
		binary.LittleEndian.PutUint16(layout.delta.bytes(destination), scaleBits)
		binary.LittleEndian.PutUint16(layout.minimum.bytes(destination), minimumBits)
		blockScale := Float16ToFloat32(scaleBits)
		blockMinimum := Float16ToFloat32(minimumBits)
		for group := 0; group < layout.block.width/32; group++ {
			quantizedScale, quantizedMinimum :=
				getKScaleMinimum(layout.scales.bytes(destination), group)
			scale := blockScale * float32(quantizedScale)
			if scale == 0 {
				continue
			}
			minimum := blockMinimum * float32(quantizedMinimum)
			for index := 0; index < 32; index++ {
				level := nearestIntGGML(
					(input[group*32+index] + minimum) / scale,
				)
				levels[group*32+index] =
					byte(max(0, min(maxLevel, level)))
			}
		}
		if dataType == dtype.Q4K {
			quantized := layout.packed.bytes(destination)
			for section := 0; section < layout.block.width; section += 64 {
				for lane := 0; lane < 32; lane++ {
					quantized[section/2+lane] =
						levels[section+lane] |
							levels[section+lane+32]<<4
				}
			}
			continue
		}
		quantized := layout.packed.bytes(destination)
		highBits := layout.high.bytes(destination)
		lowOffset := 0
		lowMask := byte(1)
		highMask := byte(2)
		for section := 0; section < layout.block.width; section += 64 {
			for lane := 0; lane < 32; lane++ {
				low := levels[section+lane]
				if low > 15 {
					low -= 16
					highBits[lane] |= lowMask
				}
				high := levels[section+lane+32]
				if high > 15 {
					high -= 16
					highBits[lane] |= highMask
				}
				quantized[lowOffset+lane] = low | high<<4
			}
			lowMask <<= 2
			highMask <<= 2
			lowOffset += 32
		}
	}
	return nil
}

func setKScaleMinimum(data []byte, group, scale, minimum int) {
	if group < 4 {
		data[group] = byte(scale)
		data[group+4] = byte(minimum)
		return
	}
	data[group+4] = byte(scale&0x0f | (minimum&0x0f)<<4)
	data[group-4] |= byte((scale >> 4) << 6)
	data[group] |= byte((minimum >> 4) << 6)
}

func getKScaleMinimum(data []byte, group int) (int, int) {
	if group < 4 {
		return int(data[group] & 63), int(data[group+4] & 63)
	}
	scale := int(data[group+4]&0x0f) |
		int(data[group-4]>>6)<<4
	minimum := int(data[group+4]>>4) |
		int(data[group]>>6)<<4
	return scale, minimum
}

func quantizeQ2K(values []float32, output []byte) error {
	layout := q2KCodec
	for block := 0; block < len(values)/layout.block.width; block++ {
		input := layout.block.input(values, block)
		destination := layout.block.storage(output, block)
		scaleMin := layout.scales.bytes(destination)
		packed := layout.packed.bytes(destination)
		levels := make([]byte, layout.block.width)
		auxiliary := make([]byte, 16)
		weights := make([]float32, 16)
		minima := make([]float32, layout.block.width/16)
		scales := make([]float32, layout.block.width/16)
		var maxScale, maxMinimum float32
		for group := 0; group < layout.block.width/16; group++ {
			groupInput := input[group*16 : (group+1)*16]
			for index, value := range groupInput {
				weights[index] = absoluteFloat32(value)
			}
			scale, minimum, err := makeQKX2Quants(
				groupInput,
				3,
				weights,
				levels[group*16:(group+1)*16],
				auxiliary,
				-0.5,
				0.1,
				15,
				true,
			)
			if err != nil {
				return fmt.Errorf("Q2_K: %w", err)
			}
			scales[group] = scale
			minima[group] = minimum
			if scale > maxScale {
				maxScale = scale
			}
			if minimum > maxMinimum {
				maxMinimum = minimum
			}
		}
		var blockScale float32
		if maxScale > 0 {
			inverse := 15 / maxScale
			for group, scale := range scales {
				scaleMin[group] = byte(nearestIntGGML(inverse * scale))
			}
			bits := Float32ToFloat16(maxScale / 15)
			binary.LittleEndian.PutUint16(layout.delta.bytes(destination), bits)
			blockScale = Float16ToFloat32(bits)
		}
		var blockMinimum float32
		if maxMinimum > 0 {
			inverse := 15 / maxMinimum
			for group, minimum := range minima {
				scaleMin[group] |=
					byte(nearestIntGGML(inverse*minimum) << 4)
			}
			bits := Float32ToFloat16(maxMinimum / 15)
			binary.LittleEndian.PutUint16(layout.minimum.bytes(destination), bits)
			blockMinimum = Float16ToFloat32(bits)
		}
		for group := 0; group < layout.block.width/16; group++ {
			scale := blockScale * float32(scaleMin[group]&0x0f)
			if scale == 0 {
				continue
			}
			minimum := blockMinimum * float32(scaleMin[group]>>4)
			for index := 0; index < 16; index++ {
				level := nearestIntGGML(
					(input[group*16+index] + minimum) / scale,
				)
				levels[group*16+index] = byte(max(0, min(3, level)))
			}
		}
		for section := 0; section < layout.block.width; section += layout.block.width / 2 {
			for lane := 0; lane < kLaneWidth; lane++ {
				packed[section/4+lane] =
					levels[section+lane] |
						levels[section+lane+kLaneWidth]<<2 |
						levels[section+lane+2*kLaneWidth]<<4 |
						levels[section+lane+3*kLaneWidth]<<6
			}
		}
	}
	return nil
}

func makeQKX2Quants(
	input []float32,
	maxLevel int,
	weights []float32,
	levels, auxiliary []byte,
	rangeMinimum, rangeDelta float32,
	steps int,
	useAbsoluteError bool,
) (float32, float32, error) {
	if len(input) == 0 ||
		len(input) != len(weights) ||
		len(input) != len(levels) ||
		len(auxiliary) < len(input) {
		return 0, 0, errors.New("invalid helper buffer lengths")
	}
	minimum := input[0]
	maximum := input[0]
	sumWeight := weights[0]
	sumValue := sumWeight * input[0]
	if !finiteFloat32(input[0]) || !finiteFloat32(weights[0]) {
		return 0, 0, errors.New("input contains a non-finite value")
	}
	for index := 1; index < len(input); index++ {
		if !finiteFloat32(input[index]) || !finiteFloat32(weights[index]) {
			return 0, 0, errors.New("input contains a non-finite value")
		}
		minimum = min(minimum, input[index])
		maximum = max(maximum, input[index])
		sumWeight += weights[index]
		sumValue += weights[index] * input[index]
	}
	if minimum > 0 {
		minimum = 0
	}
	if maximum == minimum {
		clear(levels)
		return 0, -minimum, nil
	}
	inverse := float32(maxLevel) / (maximum - minimum)
	scale := 1 / inverse
	bestError := float32(0)
	for index, value := range input {
		level := nearestIntGGML(inverse * (value - minimum))
		level = max(0, min(maxLevel, level))
		levels[index] = byte(level)
		difference := scale*float32(level) + minimum - value
		if useAbsoluteError {
			difference = absoluteFloat32(difference)
		} else {
			difference *= difference
		}
		bestError += weights[index] * difference
	}
	if steps < 1 {
		return scale, -minimum, nil
	}
	for step := 0; step <= steps; step++ {
		inverse = (rangeMinimum +
			rangeDelta*float32(step) +
			float32(maxLevel)) / (maximum - minimum)
		var sumLevel, sumLevelSquared, sumLevelValue float32
		for index, value := range input {
			level := nearestIntGGML(inverse * (value - minimum))
			level = max(0, min(maxLevel, level))
			auxiliary[index] = byte(level)
			weight := weights[index]
			sumLevel += weight * float32(level)
			sumLevelSquared += weight * float32(level*level)
			sumLevelValue += weight * float32(level) * value
		}
		determinant := sumWeight*sumLevelSquared - sumLevel*sumLevel
		if determinant <= 0 {
			continue
		}
		candidateScale :=
			(sumWeight*sumLevelValue - sumValue*sumLevel) / determinant
		candidateMinimum :=
			(sumLevelSquared*sumValue - sumLevel*sumLevelValue) /
				determinant
		if candidateMinimum > 0 {
			candidateMinimum = 0
			candidateScale = sumLevelValue / sumLevelSquared
		}
		currentError := float32(0)
		for index, value := range input {
			difference :=
				candidateScale*float32(auxiliary[index]) +
					candidateMinimum - value
			if useAbsoluteError {
				difference = absoluteFloat32(difference)
			} else {
				difference *= difference
			}
			currentError += weights[index] * difference
		}
		if currentError < bestError {
			copy(levels, auxiliary[:len(levels)])
			bestError = currentError
			scale = candidateScale
			minimum = candidateMinimum
		}
	}
	return scale, -minimum, nil
}

func quantizeQ3K(values []float32, output []byte) error {
	layout := q3KCodec
	for block := 0; block < len(values)/layout.block.width; block++ {
		input := layout.block.input(values, block)
		destination := layout.block.storage(output, block)
		scaleData := layout.scales.bytes(destination)
		highMasks := layout.high.bytes(destination)
		packed := layout.packed.bytes(destination)
		levels := make([]int8, layout.block.width)
		scales := make([]float32, layout.block.width/16)
		var maxScale, maxAbsoluteScale float32
		for group := 0; group < layout.block.width/16; group++ {
			scale, err := makeQ3Quants(
				input[group*16:(group+1)*16],
				levels[group*16:(group+1)*16],
			)
			if err != nil {
				return fmt.Errorf("Q3_K: %w", err)
			}
			scales[group] = scale
			absolute := absoluteFloat32(scale)
			if absolute > maxAbsoluteScale {
				maxAbsoluteScale = absolute
				maxScale = scale
			}
		}
		var scale float32
		if maxScale != 0 {
			inverse := -32 / maxScale
			for group := 0; group < layout.block.width/16; group++ {
				quantized := nearestIntGGML(inverse * scales[group])
				quantized = max(-32, min(31, quantized)) + 32
				if group < 8 {
					scaleData[group] = byte(quantized & 0x0f)
				} else {
					scaleData[group-8] |= byte((quantized & 0x0f) << 4)
				}
				scaleData[8+group%4] |=
					byte((quantized >> 4) << (2 * (group / 4)))
			}
			scaleBits := Float32ToFloat16(1 / inverse)
			binary.LittleEndian.PutUint16(layout.delta.bytes(destination), scaleBits)
			scale = Float16ToFloat32(scaleBits)
		}
		for group := 0; group < layout.block.width/16; group++ {
			quantizedScale := int(scaleData[group%8])
			if group < 8 {
				quantizedScale &= 0x0f
			} else {
				quantizedScale >>= 4
			}
			quantizedScale |= int(
				(scaleData[8+group%4]>>uint(2*(group/4)))&3,
			) << 4
			quantizedScale -= 32
			groupScale := scale * float32(quantizedScale)
			if groupScale == 0 {
				continue
			}
			for index := 0; index < 16; index++ {
				level := nearestIntGGML(
					input[group*16+index] / groupScale,
				)
				level = max(-4, min(3, level))
				levels[group*16+index] = int8(level + 4)
			}
		}
		maskIndex := 0
		mask := byte(1)
		for index := 0; index < layout.block.width; index++ {
			if levels[index] > 3 {
				highMasks[maskIndex] |= mask
				levels[index] -= 4
			}
			maskIndex++
			if maskIndex == layout.block.width/8 {
				maskIndex = 0
				mask <<= 1
			}
		}
		for section := 0; section < layout.block.width; section += layout.block.width / 2 {
			for lane := 0; lane < kLaneWidth; lane++ {
				packed[section/4+lane] =
					byte(levels[section+lane]) |
						byte(levels[section+lane+kLaneWidth])<<2 |
						byte(levels[section+lane+2*kLaneWidth])<<4 |
						byte(levels[section+lane+3*kLaneWidth])<<6
			}
		}
	}
	return nil
}

func makeQ3Quants(input []float32, levels []int8) (float32, error) {
	if len(input) != len(levels) {
		return 0, errors.New("input and level counts differ")
	}
	var maximum, maximumAbsolute float32
	for _, value := range input {
		if !finiteFloat32(value) {
			return 0, errors.New("input contains a non-finite value")
		}
		absolute := absoluteFloat32(value)
		if absolute > maximumAbsolute {
			maximumAbsolute = absolute
			maximum = value
		}
	}
	if maximumAbsolute < negligibleQuantizationMagnitude {
		clear(levels)
		return 0, nil
	}
	inverse := -4 / maximum
	var sumLevelValue, sumLevelSquared float32
	for index, value := range input {
		level := nearestIntGGML(inverse * value)
		level = max(-4, min(3, level))
		levels[index] = int8(level)
		weight := value * value
		sumLevelValue += weight * value * float32(level)
		sumLevelSquared += weight * float32(level*level)
	}
	for attempt := 0; attempt < 5; attempt++ {
		changed := 0
		for index, value := range input {
			weight := value * value
			oldLevel := int(levels[index])
			reducedLevelValue :=
				sumLevelValue - weight*value*float32(oldLevel)
			if reducedLevelValue <= 0 {
				continue
			}
			reducedLevelSquared :=
				sumLevelSquared - weight*float32(oldLevel*oldLevel)
			newLevel := nearestIntGGML(
				value * reducedLevelSquared / reducedLevelValue,
			)
			newLevel = max(-4, min(3, newLevel))
			if newLevel == oldLevel {
				continue
			}
			reducedLevelValue += weight * value * float32(newLevel)
			reducedLevelSquared += weight * float32(newLevel*newLevel)
			if reducedLevelSquared > 0 &&
				reducedLevelValue*reducedLevelValue*sumLevelSquared >
					sumLevelValue*sumLevelValue*reducedLevelSquared {
				levels[index] = int8(newLevel)
				sumLevelValue = reducedLevelValue
				sumLevelSquared = reducedLevelSquared
				changed++
			}
		}
		if changed == 0 {
			break
		}
	}
	for index := range levels {
		levels[index] += 4
	}
	if sumLevelSquared > 0 {
		return sumLevelValue / sumLevelSquared, nil
	}
	return 0, nil
}

func quantizeQ6K(values []float32, output []byte) error {
	layout := q6KCodec
	for block := 0; block < len(values)/layout.block.width; block++ {
		input := layout.block.input(values, block)
		destination := layout.block.storage(output, block)
		lower := layout.packed.bytes(destination)
		high := layout.high.bytes(destination)
		scaleData := layout.scales.bytes(destination)
		levels := make([]int8, layout.block.width)
		scales := make([]float32, layout.block.width/16)
		var maxScale, maxAbsoluteScale float32
		for group := 0; group < layout.block.width/16; group++ {
			scale, err := makeQXQuants(
				input[group*16:(group+1)*16],
				q6KLevelMagnitude,
				levels[group*16:(group+1)*16],
			)
			if err != nil {
				return fmt.Errorf("Q6_K: %w", err)
			}
			scales[group] = scale
			absolute := absoluteFloat32(scale)
			if absolute > maxAbsoluteScale {
				maxAbsoluteScale = absolute
				maxScale = scale
			}
		}
		if maxAbsoluteScale < negligibleQuantizationMagnitude {
			continue
		}
		inverse := -q6KScaleMagnitude / maxScale
		scaleBits := Float32ToFloat16(1 / inverse)
		binary.LittleEndian.PutUint16(layout.delta.bytes(destination), scaleBits)
		scale := Float16ToFloat32(scaleBits)
		for group := 0; group < layout.block.width/16; group++ {
			quantizedScale := min(
				q6KScaleMagnitude-1,
				nearestIntGGML(inverse*scales[group]),
			)
			scaleData[group] = byte(int8(quantizedScale))
			groupScale := scale * float32(int8(scaleData[group]))
			if groupScale == 0 {
				continue
			}
			for index := 0; index < 16; index++ {
				level := nearestIntGGML(
					input[group*16+index] / groupScale,
				)
				level = max(-q6KLevelMagnitude, min(q6KLevelMagnitude-1, level))
				levels[group*16+index] = int8(level + q6KLevelMagnitude)
			}
		}
		for section := 0; section < layout.block.width; section += layout.block.width / 2 {
			lowOffset := section / 2
			highOffset := section / 4
			for lane := 0; lane < kLaneWidth; lane++ {
				q1 := byte(levels[section+lane]) & 0x0f
				q2 := byte(levels[section+lane+kLaneWidth]) & 0x0f
				q3 := byte(levels[section+lane+2*kLaneWidth]) & 0x0f
				q4 := byte(levels[section+lane+3*kLaneWidth]) & 0x0f
				lower[lowOffset+lane] = q1 | q3<<4
				lower[lowOffset+kLaneWidth+lane] = q2 | q4<<4
				high[highOffset+lane] =
					byte(levels[section+lane])>>4 |
						(byte(levels[section+lane+kLaneWidth])>>4)<<2 |
						(byte(levels[section+lane+2*kLaneWidth])>>4)<<4 |
						(byte(levels[section+lane+3*kLaneWidth])>>4)<<6
			}
		}
	}
	return nil
}

func makeQXQuants(
	input []float32,
	maxLevel int,
	levels []int8,
) (float32, error) {
	if len(input) != len(levels) {
		return 0, errors.New("input and level counts differ")
	}
	var maximum, maximumAbsolute float32
	for _, value := range input {
		if !finiteFloat32(value) {
			return 0, errors.New("input contains a non-finite value")
		}
		absolute := absoluteFloat32(value)
		if absolute > maximumAbsolute {
			maximumAbsolute = absolute
			maximum = value
		}
	}
	if maximumAbsolute < negligibleQuantizationMagnitude {
		clear(levels)
		return 0, nil
	}
	inverse := -float32(maxLevel) / maximum
	var sumLevelValue, sumLevelSquared float32
	for index, value := range input {
		level := nearestIntGGML(inverse * value)
		level = max(-maxLevel, min(maxLevel-1, level))
		levels[index] = int8(level + maxLevel)
		weight := value * value
		sumLevelValue += weight * value * float32(level)
		sumLevelSquared += weight * float32(level*level)
	}
	scale := float32(0)
	if sumLevelSquared != 0 {
		scale = sumLevelValue / sumLevelSquared
	}
	best := scale * sumLevelValue
	for attempt := -9; attempt <= 9; attempt++ {
		if attempt == 0 {
			continue
		}
		inverse = -(float32(maxLevel) +
			float32(0.1)*float32(attempt)) / maximum
		sumLevelValue = 0
		sumLevelSquared = 0
		for _, value := range input {
			level := nearestIntGGML(inverse * value)
			level = max(-maxLevel, min(maxLevel-1, level))
			weight := value * value
			sumLevelValue += weight * value * float32(level)
			sumLevelSquared += weight * float32(level*level)
		}
		if sumLevelSquared > 0 &&
			sumLevelValue*sumLevelValue > best*sumLevelSquared {
			for index, value := range input {
				level := nearestIntGGML(inverse * value)
				level = max(-maxLevel, min(maxLevel-1, level))
				levels[index] = int8(level + maxLevel)
			}
			scale = sumLevelValue / sumLevelSquared
			best = scale * sumLevelValue
		}
	}
	return scale, nil
}

func nearestIntGGML(value float32) int {
	rounded := value + float32(12582912)
	return int(math.Float32bits(rounded)&0x007fffff) - 0x00400000
}

func quantizeQ8K(values []float32, output []byte) error {
	const (
		width    = 256
		typeSize = 292
	)
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*typeSize : (block+1)*typeSize]
		maximum, err := signedAbsoluteMaximum(input, "Q8_K")
		if err != nil {
			return err
		}
		if maximum == 0 {
			continue
		}
		inverse := -127 / maximum
		scale := 1 / inverse
		binary.LittleEndian.PutUint32(destination, math.Float32bits(scale))
		for index, value := range input {
			quantized := min(127, int(roundFloat32(inverse*value)))
			destination[4+index] = byte(int8(quantized))
		}
		for group := 0; group < width/16; group++ {
			sum := int16(0)
			for index := 0; index < 16; index++ {
				sum += int16(int8(destination[4+group*16+index]))
			}
			binary.LittleEndian.PutUint16(
				destination[260+group*2:],
				uint16(sum),
			)
		}
	}
	return nil
}

func quantizeQ1_0(values []float32, output []byte) error {
	const width = q1BlockWidth
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*q1BlockBytes : (block+1)*q1BlockBytes]
		var sumAbsolute float32
		for _, value := range input {
			if !finiteFloat32(value) {
				return errors.New("Q1_0 input contains a non-finite value")
			}
			sumAbsolute += absoluteFloat32(value)
		}
		scale := sumAbsolute / width
		binary.LittleEndian.PutUint16(destination, Float32ToFloat16(scale))
		for index, value := range input {
			if value >= 0 {
				destination[2+index/8] |= 1 << uint(index%8)
			}
		}
	}
	return nil
}

func quantizeQ2_0(values []float32, output []byte) error {
	const width = 64
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*18 : (block+1)*18]
		maximum, err := maximumAbsolute(input, "Q2_0")
		if err != nil {
			return err
		}
		inverse := float32(0)
		if maximum > 0 {
			inverse = 1 / maximum
		}
		binary.LittleEndian.PutUint16(destination, Float32ToFloat16(maximum))
		for index, value := range input {
			quantized := int(roundFloat32(value*inverse)) + 1
			quantized = max(0, min(3, quantized))
			destination[2+index/4] |= byte(quantized << uint((index%4)*2))
		}
	}
	return nil
}

func quantizeQ4(dataType dtype.Type, values []float32, output []byte) error {
	const width = 32
	typeSize := 18
	if dataType == dtype.Q4_1 {
		typeSize = 20
	}
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*typeSize : (block+1)*typeSize]
		quantizedOffset := 2
		var scale, minimum, inverse float32
		if dataType == dtype.Q4_0 {
			maximum, err := signedAbsoluteMaximum(input, "Q4_0")
			if err != nil {
				return err
			}
			scale = maximum / -8
			if scale != 0 {
				inverse = 1 / scale
			}
		} else {
			var maximum float32
			var err error
			minimum, maximum, err = minimumMaximum(input, "Q4_1")
			if err != nil {
				return err
			}
			scale = (maximum - minimum) / 15
			if scale != 0 {
				inverse = 1 / scale
			}
			binary.LittleEndian.PutUint16(
				destination[2:],
				Float32ToFloat16(minimum),
			)
			quantizedOffset = 4
		}
		binary.LittleEndian.PutUint16(destination, Float32ToFloat16(scale))
		for lane := 0; lane < width/2; lane++ {
			var low, high int
			if dataType == dtype.Q4_0 {
				low = min(15, int(input[lane]*inverse+8.5))
				high = min(15, int(input[lane+16]*inverse+8.5))
			} else {
				low = min(15, int((input[lane]-minimum)*inverse+0.5))
				high = min(15, int((input[lane+16]-minimum)*inverse+0.5))
			}
			destination[quantizedOffset+lane] = byte(low | high<<4)
		}
	}
	return nil
}

func quantizeQ5(dataType dtype.Type, values []float32, output []byte) error {
	const width = 32
	typeSize := 22
	if dataType == dtype.Q5_1 {
		typeSize = 24
	}
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*typeSize : (block+1)*typeSize]
		highOffset := 2
		var scale, minimum, inverse float32
		if dataType == dtype.Q5_0 {
			maximum, err := signedAbsoluteMaximum(input, "Q5_0")
			if err != nil {
				return err
			}
			scale = maximum / -16
			if scale != 0 {
				inverse = 1 / scale
			}
		} else {
			var maximum float32
			var err error
			minimum, maximum, err = minimumMaximum(input, "Q5_1")
			if err != nil {
				return err
			}
			scale = (maximum - minimum) / 31
			if scale != 0 {
				inverse = 1 / scale
			}
			binary.LittleEndian.PutUint16(
				destination[2:],
				Float32ToFloat16(minimum),
			)
			highOffset = 4
		}
		binary.LittleEndian.PutUint16(destination, Float32ToFloat16(scale))
		var highBits uint32
		quantizedOffset := highOffset + 4
		for lane := 0; lane < width/2; lane++ {
			var low, high int
			if dataType == dtype.Q5_0 {
				low = min(31, int(input[lane]*inverse+16.5))
				high = min(31, int(input[lane+16]*inverse+16.5))
			} else {
				low = int((input[lane]-minimum)*inverse + 0.5)
				high = int((input[lane+16]-minimum)*inverse + 0.5)
			}
			destination[quantizedOffset+lane] =
				byte(low&0x0f | (high&0x0f)<<4)
			highBits |= uint32((low>>4)&1) << uint(lane)
			highBits |= uint32((high>>4)&1) << uint(lane+16)
		}
		binary.LittleEndian.PutUint32(destination[highOffset:], highBits)
	}
	return nil
}

func quantizeQ8(dataType dtype.Type, values []float32, output []byte) error {
	const width = 32
	typeSize := 34
	if dataType == dtype.Q8_1 {
		typeSize = 36
	}
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*typeSize : (block+1)*typeSize]
		maximum, err := maximumAbsolute(input, dataType.String())
		if err != nil {
			return err
		}
		scale := maximum / 127
		inverse := float32(0)
		if scale != 0 {
			inverse = 1 / scale
		}
		binary.LittleEndian.PutUint16(destination, Float32ToFloat16(scale))
		quantizedOffset := 2
		if dataType == dtype.Q8_1 {
			quantizedOffset = 4
		}
		sum := 0
		for index, value := range input {
			quantized := int(roundFloat32(value * inverse))
			destination[quantizedOffset+index] = byte(int8(quantized))
			sum += quantized
		}
		if dataType == dtype.Q8_1 {
			binary.LittleEndian.PutUint16(
				destination[2:],
				Float32ToFloat16(float32(sum)*scale),
			)
		}
	}
	return nil
}

func quantizeMXFP4(values []float32, output []byte) error {
	const width = 32
	for block := 0; block < len(values)/width; block++ {
		input := values[block*width : (block+1)*width]
		destination := output[block*17 : (block+1)*17]
		maximum, err := maximumAbsolute(input, "MXFP4")
		if err != nil {
			return err
		}
		exponent := 0
		if maximum > 0 {
			exponent = int(math.Floor(math.Log2(float64(maximum)))) - 2 + 127
			if exponent < 0 || exponent > math.MaxUint8 {
				return errors.New("MXFP4 scale exponent is out of range")
			}
		}
		destination[0] = byte(exponent)
		var scale float32
		if exponent < 2 {
			scale = math.Float32frombits(0x00200000 << exponent)
		} else {
			scale = math.Float32frombits(uint32(exponent-1) << 23)
		}
		for lane := 0; lane < width/2; lane++ {
			low := nearestMXFP4(input[lane], scale)
			high := nearestMXFP4(input[lane+16], scale)
			destination[1+lane] = byte(low | high<<4)
		}
	}
	return nil
}

func nearestMXFP4(value, scale float32) int {
	best := 0
	bestError := absoluteFloat32(mxfp4Values[0]*scale - value)
	for index := 1; index < len(mxfp4Values); index++ {
		err := absoluteFloat32(mxfp4Values[index]*scale - value)
		if err < bestError {
			best = index
			bestError = err
		}
	}
	return best
}

func maximumAbsolute(values []float32, name string) (float32, error) {
	var maximum float32
	for _, value := range values {
		if !finiteFloat32(value) {
			return 0, fmt.Errorf("%s input contains a non-finite value", name)
		}
		maximum = max(maximum, absoluteFloat32(value))
	}
	return maximum, nil
}

func signedAbsoluteMaximum(values []float32, name string) (float32, error) {
	var absoluteMaximum float32
	var maximum float32
	for _, value := range values {
		if !finiteFloat32(value) {
			return 0, fmt.Errorf("%s input contains a non-finite value", name)
		}
		absolute := absoluteFloat32(value)
		if absoluteMaximum < absolute {
			absoluteMaximum = absolute
			maximum = value
		}
	}
	return maximum, nil
}

func minimumMaximum(values []float32, name string) (float32, float32, error) {
	minimum := float32(math.MaxFloat32)
	maximum := -float32(math.MaxFloat32)
	for _, value := range values {
		if !finiteFloat32(value) {
			return 0, 0, fmt.Errorf("%s input contains a non-finite value", name)
		}
		minimum = min(minimum, value)
		maximum = max(maximum, value)
	}
	return minimum, maximum, nil
}

func finiteFloat32(value float32) bool {
	return math.Float32bits(value)&0x7f800000 != 0x7f800000
}

func absoluteFloat32(value float32) float32 {
	return math.Float32frombits(math.Float32bits(value) & 0x7fffffff)
}

func roundFloat32(value float32) float32 {
	return float32(math.Round(float64(value)))
}

// Float32ToFloat16: binary16 round-to-nearest-even.
func Float32ToFloat16(value float32) uint16 {
	bits32 := math.Float32bits(value)
	sign := uint16(bits32>>16) & 0x8000
	exponent := int((bits32 >> 23) & 0xff)
	mantissa := bits32 & 0x007fffff

	if exponent == 0xff {
		if mantissa == 0 {
			return sign | 0x7c00
		}
		payload := uint16(mantissa >> 13)
		if payload == 0 {
			payload = 1
		}
		return sign | 0x7c00 | payload
	}

	halfExponent := exponent - 127 + 15
	if halfExponent >= 31 {
		return sign | 0x7c00
	}
	if halfExponent <= 0 {
		if halfExponent < -10 {
			return sign
		}
		mantissa |= 0x00800000
		shift := uint(14 - halfExponent)
		halfMantissa := uint16(mantissa >> shift)
		remainder := mantissa & ((uint32(1) << shift) - 1)
		halfway := uint32(1) << (shift - 1)
		if remainder > halfway ||
			remainder == halfway && halfMantissa&1 != 0 {
			halfMantissa++
		}
		return sign | halfMantissa
	}

	result := sign | uint16(halfExponent<<10) | uint16(mantissa>>13)
	remainder := mantissa & 0x1fff
	if remainder > 0x1000 || remainder == 0x1000 && result&1 != 0 {
		result++
	}
	return result
}
