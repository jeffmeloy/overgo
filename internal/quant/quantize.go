package quant

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/tensor/dtype"
)

const (
	negligibleQuantizationMagnitude         = 1e-15
	iq2MinimumMagnitude                     = 1e-8
	iq1MinimumMagnitude                     = 1e-12
	iqCodebookLaneWidth                     = 8
	iqNarrowGroupWidth                      = 2 * iqCodebookLaneWidth
	iqWideGroupWidth                        = 4 * iqCodebookLaneWidth
	iqWidePairWidth                         = 2 * iqWideGroupWidth
	iqWideSubgroupCount                     = iqWideGroupWidth / iqCodebookLaneWidth
	iq2CodebookLevelBits                    = 2
	iq2CodebookLevelMax                     = 2
	iq2CodebookStorageLevels                = 1 << iq2CodebookLevelBits
	iq2CodebookLevelMask                    = iq2CodebookStorageLevels - 1
	iq3CodebookWidth                        = 4
	iq3CodebookLevelBits                    = 3
	iq3CodebookLevelMax                     = 1<<iq3CodebookLevelBits - 1
	minimumQuantizedLevel                   = 0
	maximumAffineMinimum            float32 = 0
	nearestDistanceTierCount                = 1
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
	blocks, aligned := traits.BlockCount(uint64(len(values)))
	if !aligned {
		return nil, fmt.Errorf(
			"element count %d is not divisible by %s block size %d",
			len(values),
			traits.Name,
			traits.BlockSize,
		)
	}
	if blocks > uint64(math.MaxInt)/traits.TypeSize {
		return nil, errors.New("quantized tensor exceeds addressable memory")
	}
	output := make([]byte, int(blocks*traits.TypeSize))
	switch dataType {
	case dtype.F32:
		for index, value := range values {
			binary.LittleEndian.PutUint32(output[index*binaryschema.Uint32Bytes:], math.Float32bits(value))
		}
	case dtype.F16:
		for index, value := range values {
			binary.LittleEndian.PutUint16(output[index*binaryschema.Uint16Bytes:], Float32ToFloat16(value))
		}
	case dtype.BF16:
		for index, value := range values {
			binary.LittleEndian.PutUint16(output[index*binaryschema.Uint16Bytes:], dtype.Float32ToBF16(value))
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
		if err := quantizeQ4Or5(dataType, values, output); err != nil {
			return nil, err
		}
	case dtype.Q5_0, dtype.Q5_1:
		if err := quantizeQ4Or5(dataType, values, output); err != nil {
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

type iqQuantCodebook struct {
	width     int
	levelBits uint
	lanes     []int8
	index     map[uint16]int
}

func (codebook iqQuantCodebook) lane(index int) []int8 {
	return codebook.lanes[index*codebook.width : (index+1)*codebook.width]
}

type iqPackedGrid interface {
	~uint32 | ~uint64
}

var (
	iq3XXSQuantCodebook = buildIQQuantCodebook(iq3XXSGrid[:], iq3CodebookWidth, 3)
	iq3SQuantCodebook   = buildIQQuantCodebook(iq3SGrid[:], iq3CodebookWidth, 3)
)

func buildIQQuantCodebook[T iqPackedGrid](grid []T, width int, levelBits uint) iqQuantCodebook {
	unique := make([]byte, 0, width)
	for _, packed := range grid {
		for lane := 0; lane < width; lane++ {
			value := byte(packed >> uint(lane*8))
			if !slices.Contains(unique, value) {
				unique = append(unique, value)
			}
		}
	}
	slices.Sort(unique)
	result := iqQuantCodebook{
		width: width, levelBits: levelBits,
		lanes: make([]int8, len(grid)*width), index: make(map[uint16]int, len(grid)),
	}
	for gridIndex, packed := range grid {
		var encoded uint16
		for lane := 0; lane < width; lane++ {
			value := byte(packed >> uint(lane*8))
			level, _ := slices.BinarySearch(unique, value)
			result.lanes[gridIndex*width+lane] = int8(2*level + 1)
			encoded |= uint16(level) << uint(levelBits*uint(lane))
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
	layout := iq3XXSBlockLayout
	codebook := iq3XXSQuantCodebook
	distanceTiers := 2
	attempts := 15
	attemptStep := float32(0.2)
	paritySigns := true
	refineAll := false
	scaleFudge := float32(1.0125)
	if dataType == dtype.IQ3S {
		layout = iq3SBlockLayout
		codebook = iq3SQuantCodebook
		distanceTiers = 3
		attempts = 9
		paritySigns = false
		refineAll = true
		scaleFudge = 1.033
	}
	scales := make([]float32, layout.elements/iqWideGroupWidth)
	for block := 0; block < len(values)/layout.elements; block++ {
		input := layout.input(values, block)
		destination := layout.storage(output, block)
		for _, value := range input {
			if !finiteFloat32(value) {
				return fmt.Errorf("%s input contains a non-finite value", dataType)
			}
		}
		var maxScale float32
		for group := 0; group < layout.elements/iqWideGroupWidth; group++ {
			indices, signs, scale, err := quantizeIQ3Group(
				input[group*iqWideGroupWidth:(group+1)*iqWideGroupWidth],
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
			quantized = max(minimumQuantizedLevel, min(binaryschema.NibbleMask, quantized))
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
	codebook iqQuantCodebook,
	distanceTiers, attempts int,
	attemptStep float32,
	paritySigns, refineAll bool,
) ([iqWideGroupWidth / iq3CodebookWidth]int, [iqWideGroupWidth / iqCodebookLaneWidth]byte, float32, error) {
	var indices [iqWideGroupWidth / iq3CodebookWidth]int
	var signs [iqWideGroupWidth / iqCodebookLaneWidth]byte
	weight := make([]float32, iqWideGroupWidth)
	neighborWeight := make([]float32, iqWideGroupWidth)
	absoluteValues := make([]float32, iqWideGroupWidth)
	levels := make([]int8, iqWideGroupWidth)
	auxiliary := make([]int8, iqWideGroupWidth)
	onGrid := [iqWideGroupWidth / iq3CodebookWidth]bool{}
	auxiliaryOnGrid := [iqWideGroupWidth / iq3CodebookWidth]bool{}
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
			signs[signGroup] &= iqSignPayloadMask
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
	scale := maximum / binaryschema.NibbleMask
	for attempt := -attempts; attempt <= attempts; attempt++ {
		inverse := (binaryschema.NibbleMask + float32(attempt)*attemptStep) / maximum
		candidateScale := 1 / inverse
		for subGroup := range indices {
			var encoded uint16
			for lane := 0; lane < iq3CodebookWidth; lane++ {
				index := subGroup*iq3CodebookWidth + lane
				level := nearestIntGGML(0.5 *
					(inverse*absoluteValues[index] - 1))
				level = max(minimumQuantizedLevel, min(iq3CodebookLevelMax, level))
				auxiliary[index] = int8(level)
				encoded |= uint16(level) << uint(iq3CodebookLevelBits*lane)
			}
			_, direct := codebook.index[encoded]
			auxiliaryOnGrid[subGroup] = direct
			if !direct {
				iqFindBest(
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
		for index := 0; index < iqWideGroupWidth; index++ {
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
				level = max(minimumQuantizedLevel, min(iq3CodebookLevelMax, level))
				encoded |= uint16(level) << uint(iq3CodebookLevelBits*lane)
			}
			if gridIndex, direct := codebook.index[encoded]; direct {
				for lane, value := range codebook.lane(gridIndex) {
					levels[subGroup*iq3CodebookWidth+lane] = (value - 1) / 2
				}
			} else {
				iqFindBest(
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
		for index := 0; index < iqWideGroupWidth; index++ {
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
				signs[index] &= iqSignPayloadMask
			}
		}
	}
	for subGroup := range indices {
		var encoded uint16
		for lane := 0; lane < iq3CodebookWidth; lane++ {
			encoded |= uint16(levels[subGroup*iq3CodebookWidth+lane]) << uint(iq3CodebookLevelBits*lane)
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

func iqFindBest(
	codebook iqQuantCodebook,
	encoded uint16,
	values, weights []float32,
	scale float32,
	levels []int8,
	distanceTiers int,
) int {
	var targetStorage [iqCodebookLaneWidth]int8
	target := targetStorage[:codebook.width]
	mask := uint16(1<<codebook.levelBits) - 1
	for lane := range target {
		target[lane] = 2*int8((encoded>>uint(codebook.levelBits*uint(lane)))&mask) + 1
	}
	gridCount := len(codebook.lanes) / codebook.width
	tiers := make([]int, 0, min(distanceTiers, gridCount))
	distance := func(grid []int8) int {
		distance := 0
		for lane := range target {
			difference := int(grid[lane] - target[lane])
			distance += difference * difference
		}
		return distance
	}
	for index := 0; index < gridCount; index++ {
		value := distance(codebook.lane(index))
		position, present := slices.BinarySearch(tiers, value)
		if present || position >= distanceTiers {
			continue
		}
		if len(tiers) < distanceTiers {
			tiers = append(tiers, 0)
		}
		copy(tiers[position+1:], tiers[position:])
		tiers[position] = value
		if len(tiers) > distanceTiers {
			tiers = tiers[:distanceTiers]
		}
	}
	threshold := tiers[len(tiers)-1]
	bestError := float32(math.MaxFloat32)
	bestIndex := -1
	for index := 0; index < gridCount; index++ {
		grid := codebook.lane(index)
		if distance(grid) > threshold {
			continue
		}
		currentError := float32(0)
		for lane := range target {
			difference := scale*float32(grid[lane]) - values[lane]
			currentError += weights[lane] * difference * difference
		}
		if currentError < bestError {
			bestError = currentError
			bestIndex = index
		}
	}
	for lane, value := range codebook.lane(bestIndex) {
		levels[lane] = (value - 1) / 2
	}
	return bestIndex
}

var iq2SQuantCodebook = buildIQQuantCodebook(iq2SGrid[:], iqCodebookLaneWidth, iq2CodebookLevelBits)

func quantizeIQ2S(values []float32, output []byte) error {
	layout := iq2SBlockLayout
	scales := make([]float32, layout.elements/iqNarrowGroupWidth)
	weight := make([]float32, iqNarrowGroupWidth)
	neighborWeight := make([]float32, iqNarrowGroupWidth)
	absoluteValues := make([]float32, iqNarrowGroupWidth)
	levels := make([]int8, iqNarrowGroupWidth)
	auxiliary := make([]int8, iqNarrowGroupWidth)
	for block := 0; block < len(values)/layout.elements; block++ {
		input := layout.input(values, block)
		destination := layout.storage(output, block)
		sumSquares := float32(0)
		for _, value := range input {
			if !finiteFloat32(value) {
				return errors.New("IQ2_S input contains a non-finite value")
			}
			sumSquares += value * value
		}
		variance := 2 * sumSquares / float32(layout.elements)
		var maxScale float32
		for group := 0; group < layout.elements/iqNarrowGroupWidth; group++ {
			groupInput := input[group*iqNarrowGroupWidth : (group+1)*iqNarrowGroupWidth]
			clear(levels)
			clear(auxiliary)
			onGrid := [iqNarrowGroupWidth / iqCodebookLaneWidth]bool{true, true}
			auxiliaryOnGrid := [iqNarrowGroupWidth / iqCodebookLaneWidth]bool{}
			signs := [iqNarrowGroupWidth / iqCodebookLaneWidth]byte{}
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
						level = max(minimumQuantizedLevel, min(iq2CodebookLevelMax, level))
						auxiliary[subGroup*iqCodebookLaneWidth+lane] = int8(level)
						encoded |= uint16(level) << uint(iq2CodebookLevelBits*lane)
					}
					_, direct := iq2SQuantCodebook.index[encoded]
					auxiliaryOnGrid[subGroup] = direct
					if !direct {
						iqFindBest(
							iq2SQuantCodebook,
							encoded,
							absoluteValues[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							neighborWeight[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							candidateScale,
							auxiliary[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							nearestDistanceTierCount,
						)
					}
				}
				var sumValue, sumQuantized float32
				for index := 0; index < iqNarrowGroupWidth; index++ {
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
						level = max(minimumQuantizedLevel, min(iq2CodebookLevelMax, level))
						levels[subGroup*iqCodebookLaneWidth+lane] = int8(level)
						encoded |= uint16(level) << uint(iq2CodebookLevelBits*lane)
					}
					if _, direct := iq2SQuantCodebook.index[encoded]; !direct {
						iqFindBest(
							iq2SQuantCodebook,
							encoded,
							absoluteValues[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							neighborWeight[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							scale,
							levels[subGroup*iqCodebookLaneWidth:(subGroup+1)*iqCodebookLaneWidth],
							nearestDistanceTierCount,
						)
					}
				}
				var sumValue, sumQuantized float32
				for index := 0; index < iqNarrowGroupWidth; index++ {
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
				destination[iq2SGridStart+index] = byte(gridIndex)
				destination[iq2SHighStart+index/4] |=
					byte(gridIndex>>8) << uint(2*(index%4))
				destination[iq2SSignStart+index] = signs[subGroup]
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
			quantized = max(minimumQuantizedLevel, min(binaryschema.NibbleMask, quantized))
			if group%2 == 0 {
				destination[iq2SScaleStart+group/2] = byte(quantized)
			} else {
				destination[iq2SScaleStart+group/2] |= byte(quantized << 4)
			}
		}
	}
	return nil
}

func encodeIQ2Levels(levels []int8) uint16 {
	var encoded uint16
	for lane, level := range levels {
		encoded |= uint16(level) << uint(iq2CodebookLevelBits*lane)
	}
	return encoded
}

func quantizeIQ4(
	dataType dtype.Type,
	values []float32,
	output []byte,
) error {
	groupWidth := iq4NLBlockLayout.elements
	layout := iq4NLBlockLayout
	quantizedOffset := 2
	attempts := -1
	if dataType == dtype.IQ4XS {
		layout = iq4XSBlockLayout
		quantizedOffset = 8
		attempts = 7
	}
	levels := make([]byte, layout.elements)
	scales := make([]float32, layout.elements/groupWidth)
	for block := 0; block < len(values)/layout.elements; block++ {
		input := layout.input(values, block)
		destination := layout.storage(output, block)
		var maxScale, maxAbsoluteScale float32
		for group := 0; group < layout.elements/groupWidth; group++ {
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
			for group := 0; group < layout.elements/groupWidth; group++ {
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
					destination[4+group/2] = byte(encodedScale & binaryschema.NibbleMask)
				} else {
					destination[4+group/2] |= byte(
						(encodedScale & binaryschema.NibbleMask) << binaryschema.NibbleBits,
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
		for group := 0; group < layout.elements/groupWidth; group++ {
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
	layout := nvfp4BlockLayout
	subBlockWidth := layout.elements / nvfp4ScaleCount
	packedSubBlockBytes := subBlockWidth / 2
	for block := 0; block < len(values)/layout.elements; block++ {
		input := layout.input(values, block)
		destination := layout.storage(output, block)
		for subBlock := 0; subBlock < nvfp4ScaleCount; subBlock++ {
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
				destination[nvfp4PackedStart+subBlock*packedSubBlockBytes+lane] = byte(low | high<<4)
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
	layout := tq1BlockLayout
	if dataType == dtype.TQ2_0 {
		layout = tq2BlockLayout
	}
	for block := 0; block < len(values)/layout.elements; block++ {
		input := layout.input(values, block)
		destination := layout.storage(output, block)
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
			for section := 0; section < layout.elements; section += tq2SectionWidth {
				for lane := 0; lane < tq2LaneWidth; lane++ {
					quantized := byte(0)
					for group := 0; group < tq2GroupCount; group++ {
						level := int(roundFloat32(
							input[section+lane+group*tq2LaneWidth]*inverse,
						)) + 1
						quantized |= byte(level&3) << uint(2*group)
					}
					destination[section/4+lane] = quantized
				}
			}
			continue
		}
		for lane := 0; lane < tq1WideLaneWidth; lane++ {
			quantized := byte(0)
			for group := 0; group < tq1MainTritCount; group++ {
				level := int(roundFloat32(
					input[lane+group*tq1WideLaneWidth]*inverse,
				)) + 1
				quantized = quantized*3 + byte(level)
			}
			destination[lane] = byte(
				(uint16(quantized)*256 + 242) / 243,
			)
		}
		for lane := 0; lane < tq1NarrowLaneWidth; lane++ {
			quantized := byte(0)
			for group := 0; group < tq1MainTritCount; group++ {
				level := int(roundFloat32(
					input[tq1NarrowInputStart+lane+group*tq1NarrowLaneWidth]*inverse,
				)) + 1
				quantized = quantized*3 + byte(level)
			}
			destination[tq1NarrowPackedStart+lane] = byte(
				(uint16(quantized)*256 + 242) / 243,
			)
		}
		for lane := 0; lane < tq1TailLaneWidth; lane++ {
			quantized := byte(0)
			for group := 0; group < tq1TailTritCount; group++ {
				level := int(roundFloat32(
					input[tq1TailInputStart+lane+group*tq1TailLaneWidth]*inverse,
				)) + 1
				quantized = quantized*3 + byte(level)
			}
			quantized *= 3
			destination[tq1TailPackedStart+lane] = byte(
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
	if dataType == dtype.Q5K {
		layout = q5KCodec
	}
	levels := make([]byte, layout.block.elements)
	auxiliary := make([]byte, layout.group.width)
	weights := make([]float32, layout.group.width)
	minima := make([]float32, layout.groupCount())
	scales := make([]float32, layout.groupCount())
	for block := 0; block < len(values)/layout.block.elements; block++ {
		input := layout.block.input(values, block)
		destination := layout.block.storage(output, block)
		var maxScale, maxMinimum float32
		for group := 0; group < layout.groupCount(); group++ {
			clear(auxiliary)
			groupInput := layout.groupInput(input, group)
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
			average := float32(math.Sqrt(float64(sumSquares / float32(layout.group.width))))
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
				layout.group.levelMax,
				weights,
				layout.groupLevels(levels, group),
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
			inverseScale = float32(layout.group.scaleMax) / maxScale
		}
		inverseMinimum := float32(0)
		if maxMinimum > 0 {
			inverseMinimum = float32(layout.group.scaleMax) / maxMinimum
		}
		for group := 0; group < layout.groupCount(); group++ {
			scale := min(layout.group.scaleMax, nearestIntGGML(inverseScale*scales[group]))
			minimum := min(
				layout.group.scaleMax,
				nearestIntGGML(inverseMinimum*minima[group]),
			)
			layout.setScaleMinimum(
				layout.scales.bytes(destination),
				group,
				scale,
				minimum,
			)
		}
		scaleBits := Float32ToFloat16(maxScale / float32(layout.group.scaleMax))
		minimumBits := Float32ToFloat16(maxMinimum / float32(layout.group.scaleMax))
		binary.LittleEndian.PutUint16(layout.delta.bytes(destination), scaleBits)
		binary.LittleEndian.PutUint16(layout.minimum.bytes(destination), minimumBits)
		blockScale := Float16ToFloat32(scaleBits)
		blockMinimum := Float16ToFloat32(minimumBits)
		for group := 0; group < layout.groupCount(); group++ {
			quantizedScale, quantizedMinimum :=
				layout.scaleMinimum(layout.scales.bytes(destination), group)
			scale := blockScale * float32(quantizedScale)
			if scale == 0 {
				continue
			}
			minimum := blockMinimum * float32(quantizedMinimum)
			groupLevels := layout.groupLevels(levels, group)
			groupInput := layout.groupInput(input, group)
			for index := range groupLevels {
				level := nearestIntGGML(
					(groupInput[index] + minimum) / scale,
				)
				groupLevels[index] = byte(max(minimumQuantizedLevel, min(layout.group.levelMax, level)))
			}
		}
		if dataType == dtype.Q4K {
			quantized := layout.packed.bytes(destination)
			sectionWidth := 2 * layout.group.width
			for section := 0; section < layout.block.elements; section += sectionWidth {
				for lane := 0; lane < layout.group.width; lane++ {
					quantized[section/2+lane] =
						levels[section+lane] |
							levels[section+lane+layout.group.width]<<layout.group.packedBits
				}
			}
			continue
		}
		quantized := layout.packed.bytes(destination)
		highBits := layout.high.bytes(destination)
		lowOffset := 0
		groupsPerSection := binaryschema.BitsPerByte / int(layout.group.packedBits)
		sectionWidth := groupsPerSection * layout.group.width
		lowMask := byte(1)
		highMask := lowMask << (groupsPerSection - 1)
		lowLevelCount := byte(1 << layout.group.packedBits)
		for section := 0; section < layout.block.elements; section += sectionWidth {
			for lane := 0; lane < layout.group.width; lane++ {
				low := levels[section+lane]
				if low >= lowLevelCount {
					low -= lowLevelCount
					highBits[lane] |= lowMask
				}
				high := levels[section+lane+layout.group.width]
				if high >= lowLevelCount {
					high -= lowLevelCount
					highBits[lane] |= highMask
				}
				quantized[lowOffset+lane] = low | high<<layout.group.packedBits
			}
			lowMask <<= groupsPerSection
			highMask <<= groupsPerSection
			lowOffset += layout.group.width
		}
	}
	return nil
}

func quantizeQ2K(values []float32, output []byte) error {
	layout := q2KCodec
	levels := make([]byte, layout.block.elements)
	auxiliary := make([]byte, layout.group.width)
	weights := make([]float32, layout.group.width)
	minima := make([]float32, layout.groupCount())
	scales := make([]float32, layout.groupCount())
	for block := 0; block < len(values)/layout.block.elements; block++ {
		input := layout.block.input(values, block)
		destination := layout.block.storage(output, block)
		scaleMin := layout.scales.bytes(destination)
		packed := layout.packed.bytes(destination)
		var maxScale, maxMinimum float32
		for group := 0; group < layout.groupCount(); group++ {
			clear(auxiliary)
			groupInput := layout.groupInput(input, group)
			for index, value := range groupInput {
				weights[index] = absoluteFloat32(value)
			}
			scale, minimum, err := makeQKX2Quants(
				groupInput,
				layout.group.levelMax,
				weights,
				layout.groupLevels(levels, group),
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
			inverse := float32(layout.group.scaleMax) / maxScale
			for group, scale := range scales {
				scaleMin[group] = byte(nearestIntGGML(inverse * scale))
			}
			bits := Float32ToFloat16(maxScale / float32(layout.group.scaleMax))
			binary.LittleEndian.PutUint16(layout.delta.bytes(destination), bits)
			blockScale = Float16ToFloat32(bits)
		}
		var blockMinimum float32
		if maxMinimum > 0 {
			inverse := float32(layout.group.scaleMax) / maxMinimum
			for group, minimum := range minima {
				scaleMin[group] |=
					byte(nearestIntGGML(inverse*minimum) << layout.group.scaleBits)
			}
			bits := Float32ToFloat16(maxMinimum / float32(layout.group.scaleMax))
			binary.LittleEndian.PutUint16(layout.minimum.bytes(destination), bits)
			blockMinimum = Float16ToFloat32(bits)
		}
		for group := 0; group < layout.groupCount(); group++ {
			scale := blockScale * float32(scaleMin[group]&layout.group.scaleMask())
			if scale == 0 {
				continue
			}
			minimum := blockMinimum * float32(scaleMin[group]>>layout.group.scaleBits)
			groupInput := layout.groupInput(input, group)
			groupLevels := layout.groupLevels(levels, group)
			for index := range groupLevels {
				level := nearestIntGGML(
					(groupInput[index] + minimum) / scale,
				)
				groupLevels[index] = byte(max(minimumQuantizedLevel, min(layout.group.levelMax, level)))
			}
		}
		for section := 0; section < layout.block.elements; section += layout.block.elements / 2 {
			for lane := 0; lane < kLaneWidth; lane++ {
				packed[section/4+lane] =
					levels[section+lane] |
						levels[section+lane+kLaneWidth]<<layout.group.packedBits |
						levels[section+lane+2*kLaneWidth]<<(2*layout.group.packedBits) |
						levels[section+lane+3*kLaneWidth]<<(3*layout.group.packedBits)
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
	if minimum > maximumAffineMinimum {
		minimum = maximumAffineMinimum
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
		level = max(minimumQuantizedLevel, min(maxLevel, level))
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
			level = max(minimumQuantizedLevel, min(maxLevel, level))
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
		if candidateMinimum > maximumAffineMinimum {
			candidateMinimum = maximumAffineMinimum
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
	levels := make([]int8, layout.block.elements)
	scales := make([]float32, layout.groupCount())
	for block := 0; block < len(values)/layout.block.elements; block++ {
		input := layout.block.input(values, block)
		destination := layout.block.storage(output, block)
		scaleData := layout.scales.bytes(destination)
		highMasks := layout.high.bytes(destination)
		packed := layout.packed.bytes(destination)
		var maxScale, maxAbsoluteScale float32
		for group := 0; group < layout.groupCount(); group++ {
			scale, err := makeQ3Quants(
				layout.groupInput(input, group),
				layout.groupLevels8(levels, group),
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
			inverse := -float32(layout.group.scaleZero) / maxScale
			for group := 0; group < layout.groupCount(); group++ {
				quantized := nearestIntGGML(inverse * scales[group])
				quantized = max(-layout.group.scaleZero, min(layout.group.scaleMax, quantized)) +
					layout.group.scaleZero
				layout.setScaleLevel(scaleData, group, quantized)
			}
			scaleBits := Float32ToFloat16(1 / inverse)
			binary.LittleEndian.PutUint16(layout.delta.bytes(destination), scaleBits)
			scale = Float16ToFloat32(scaleBits)
		}
		for group := 0; group < layout.groupCount(); group++ {
			quantizedScale := layout.scaleLevel(scaleData, group)
			groupScale := scale * float32(quantizedScale)
			if groupScale == 0 {
				continue
			}
			groupInput := layout.groupInput(input, group)
			groupLevels := layout.groupLevels8(levels, group)
			minimumLevel, maximumLevel := layout.group.centeredLevelBounds()
			for index := range groupLevels {
				level := nearestIntGGML(
					groupInput[index] / groupScale,
				)
				level = max(minimumLevel, min(maximumLevel, level))
				groupLevels[index] = int8(level + layout.group.levelZero)
			}
		}
		maskIndex := 0
		mask := byte(1)
		for index := 0; index < layout.block.elements; index++ {
			if int(levels[index]) > layout.group.packedLevelMax() {
				highMasks[maskIndex] |= mask
				levels[index] -= 1 << layout.group.packedBits
			}
			maskIndex++
			if maskIndex == layout.block.elements/binaryschema.BitsPerByte {
				maskIndex = 0
				mask <<= 1
			}
		}
		for section := 0; section < layout.block.elements; section += layout.block.elements / 2 {
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
	levels := make([]int8, layout.block.elements)
	scales := make([]float32, layout.groupCount())
	for block := 0; block < len(values)/layout.block.elements; block++ {
		input := layout.block.input(values, block)
		destination := layout.block.storage(output, block)
		lower := layout.packed.bytes(destination)
		high := layout.high.bytes(destination)
		scaleData := layout.scales.bytes(destination)
		var maxScale, maxAbsoluteScale float32
		for group := 0; group < layout.groupCount(); group++ {
			scale, err := makeQXQuants(
				layout.groupInput(input, group),
				layout.group.levelZero,
				layout.groupLevels8(levels, group),
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
		inverse := -float32(layout.group.scaleZero) / maxScale
		scaleBits := Float32ToFloat16(1 / inverse)
		binary.LittleEndian.PutUint16(layout.delta.bytes(destination), scaleBits)
		scale := Float16ToFloat32(scaleBits)
		for group := 0; group < layout.groupCount(); group++ {
			quantizedScale := min(
				layout.group.scaleMax,
				nearestIntGGML(inverse*scales[group]),
			)
			scaleData[group] = byte(int8(quantizedScale))
			groupScale := scale * float32(int8(scaleData[group]))
			if groupScale == 0 {
				continue
			}
			groupInput := layout.groupInput(input, group)
			groupLevels := layout.groupLevels8(levels, group)
			minimumLevel, maximumLevel := layout.group.centeredLevelBounds()
			for index := range groupLevels {
				level := nearestIntGGML(
					groupInput[index] / groupScale,
				)
				level = max(minimumLevel, min(maximumLevel, level))
				groupLevels[index] = int8(level + layout.group.levelZero)
			}
		}
		for section := 0; section < layout.block.elements; section += layout.block.elements / 2 {
			lowOffset := section / 2
			highOffset := section / 4
			for lane := 0; lane < kLaneWidth; lane++ {
				q1 := byte(levels[section+lane]) & binaryschema.NibbleMask
				q2 := byte(levels[section+lane+kLaneWidth]) & binaryschema.NibbleMask
				q3 := byte(levels[section+lane+2*kLaneWidth]) & binaryschema.NibbleMask
				q4 := byte(levels[section+lane+3*kLaneWidth]) & binaryschema.NibbleMask
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
	codec := scalarCodecs[dtype.Q8K]
	for block := 0; block < len(values)/codec.block.elements; block++ {
		input := codec.block.input(values, block)
		destination := codec.block.storage(output, block)
		maximum, err := signedAbsoluteMaximum(input, "Q8_K")
		if err != nil {
			return err
		}
		if maximum == 0 {
			continue
		}
		inverse := -127 / maximum
		scale := 1 / inverse
		binary.LittleEndian.PutUint32(codec.scale.bytes(destination), math.Float32bits(scale))
		packed := codec.packed.bytes(destination)
		for index, value := range input {
			quantized := min(127, int(roundFloat32(inverse*value)))
			packed[index] = byte(int8(quantized))
		}
		sums := codec.tail.bytes(destination)
		for group := 0; group < codec.block.elements/codec.tailGroup; group++ {
			sum := int16(0)
			for index := 0; index < codec.tailGroup; index++ {
				sum += int16(int8(packed[group*codec.tailGroup+index]))
			}
			binary.LittleEndian.PutUint16(
				sums[group*binaryschema.Uint16Bytes:],
				uint16(sum),
			)
		}
	}
	return nil
}

func quantizeQ1_0(values []float32, output []byte) error {
	codec := scalarCodecs[dtype.Q1_0]
	for block := 0; block < len(values)/codec.block.elements; block++ {
		input := codec.block.input(values, block)
		destination := codec.block.storage(output, block)
		var sumAbsolute float32
		for _, value := range input {
			if !finiteFloat32(value) {
				return errors.New("Q1_0 input contains a non-finite value")
			}
			sumAbsolute += absoluteFloat32(value)
		}
		scale := sumAbsolute / float32(codec.block.elements)
		binary.LittleEndian.PutUint16(codec.scale.bytes(destination), Float32ToFloat16(scale))
		packed := codec.packed.bytes(destination)
		for index, value := range input {
			if value >= 0 {
				packed[index/8] |= 1 << uint(index%8)
			}
		}
	}
	return nil
}

func quantizeQ2_0(values []float32, output []byte) error {
	codec := scalarCodecs[dtype.Q2_0]
	for block := 0; block < len(values)/codec.block.elements; block++ {
		input := codec.block.input(values, block)
		destination := codec.block.storage(output, block)
		maximum, err := maximumAbsolute(input, "Q2_0")
		if err != nil {
			return err
		}
		inverse := float32(0)
		if maximum > 0 {
			inverse = 1 / maximum
		}
		binary.LittleEndian.PutUint16(codec.scale.bytes(destination), Float32ToFloat16(maximum))
		packed := codec.packed.bytes(destination)
		for index, value := range input {
			quantized := int(roundFloat32(value*inverse)) + 1
			quantized = max(minimumQuantizedLevel, min(3, quantized))
			packed[index/4] |= byte(quantized << uint((index%4)*2))
		}
	}
	return nil
}

func quantizeQ4Or5(dataType dtype.Type, values []float32, output []byte) error {
	codec := scalarCodecs[dataType]
	for block := 0; block < len(values)/codec.block.elements; block++ {
		input := codec.block.input(values, block)
		destination := codec.block.storage(output, block)
		var scale, minimum, inverse float32
		zeroPoint := codec.zeroPoint
		if zeroPoint != 0 {
			maximum, err := signedAbsoluteMaximum(input, dataType.String())
			if err != nil {
				return err
			}
			scale = maximum / -float32(zeroPoint)
			if scale != 0 {
				inverse = 1 / scale
			}
		} else {
			var maximum float32
			var err error
			minimum, maximum, err = minimumMaximum(input, dataType.String())
			if err != nil {
				return err
			}
			scale = (maximum - minimum) / float32(codec.levels)
			if scale != 0 {
				inverse = 1 / scale
			}
			binary.LittleEndian.PutUint16(codec.minimum.bytes(destination), Float32ToFloat16(minimum))
		}
		binary.LittleEndian.PutUint16(codec.scale.bytes(destination), Float32ToFloat16(scale))
		packed := codec.packed.bytes(destination)
		packedMask := 1<<codec.packedBits - 1
		rounding := float32(zeroPoint) + 0.5
		half := codec.block.elements / 2
		var highBits uint32
		for lane := 0; lane < half; lane++ {
			var low, high int
			if zeroPoint != 0 {
				low = min(codec.levels, int(input[lane]*inverse+rounding))
				high = min(codec.levels, int(input[lane+half]*inverse+rounding))
			} else {
				low = int((input[lane]-minimum)*inverse + rounding)
				high = int((input[lane+half]-minimum)*inverse + rounding)
				if !codec.high.present() {
					low, high = min(codec.levels, low), min(codec.levels, high)
				}
			}
			packed[lane] = byte(low&packedMask | (high&packedMask)<<codec.packedBits)
			if codec.high.present() {
				highBits |= uint32((low>>codec.packedBits)&1) << uint(lane)
				highBits |= uint32((high>>codec.packedBits)&1) << uint(lane+half)
			}
		}
		if codec.high.present() {
			binary.LittleEndian.PutUint32(codec.high.bytes(destination), highBits)
		}
	}
	return nil
}

func quantizeQ8(dataType dtype.Type, values []float32, output []byte) error {
	codec := scalarCodecs[dataType]
	for block := 0; block < len(values)/codec.block.elements; block++ {
		input := codec.block.input(values, block)
		destination := codec.block.storage(output, block)
		maximum, err := maximumAbsolute(input, dataType.String())
		if err != nil {
			return err
		}
		scale := maximum / 127
		inverse := float32(0)
		if scale != 0 {
			inverse = 1 / scale
		}
		binary.LittleEndian.PutUint16(codec.scale.bytes(destination), Float32ToFloat16(scale))
		packed := codec.packed.bytes(destination)
		sum := 0
		for index, value := range input {
			quantized := int(roundFloat32(value * inverse))
			packed[index] = byte(int8(quantized))
			sum += quantized
		}
		if dataType == dtype.Q8_1 {
			binary.LittleEndian.PutUint16(codec.minimum.bytes(destination), Float32ToFloat16(float32(sum)*scale))
		}
	}
	return nil
}

func quantizeMXFP4(values []float32, output []byte) error {
	layout := mxfp4BlockLayout
	for block := 0; block < len(values)/layout.elements; block++ {
		input := layout.input(values, block)
		destination := layout.storage(output, block)
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
		for lane := 0; lane < layout.elements/2; lane++ {
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
