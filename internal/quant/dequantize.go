package quant

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor/dtype"
)

// Dequantize: converts supported ggml storage into F32 values
func Dequantize(dataType dtype.Type, source []byte, elements uint64) ([]float32, error) {
	if elements > uint64(math.MaxInt) {
		return nil, errors.New("dequantized tensor exceeds addressable memory")
	}
	traits, ok := dataType.Traits()
	if !ok {
		return nil, fmt.Errorf("unknown tensor type %d", dataType)
	}
	if elements%traits.BlockSize != 0 {
		return nil, fmt.Errorf("element count %d is not divisible by %s block size %d", elements, traits.Name, traits.BlockSize)
	}
	blocks := elements / traits.BlockSize
	if blocks > math.MaxUint64/traits.TypeSize {
		return nil, errors.New("quantized byte size overflows uint64")
	}
	expected := blocks * traits.TypeSize
	if uint64(len(source)) != expected {
		return nil, fmt.Errorf("%s data has %d bytes, need %d", traits.Name, len(source), expected)
	}

	output := make([]float32, int(elements))
	switch dataType {
	case dtype.F32:
		for index := range output {
			output[index] = math.Float32frombits(binary.LittleEndian.Uint32(source[index*4:]))
		}
	case dtype.F16:
		for index := range output {
			output[index] = Float16ToFloat32(binary.LittleEndian.Uint16(source[index*2:]))
		}
	case dtype.BF16:
		for index := range output {
			output[index] = dtype.BF16ToFloat32(binary.LittleEndian.Uint16(source[index*2:]))
		}
	case dtype.I8:
		for index := range output {
			output[index] = float32(int8(source[index]))
		}
	case dtype.I16:
		for index := range output {
			output[index] = float32(int16(binary.LittleEndian.Uint16(source[index*2:])))
		}
	case dtype.I32:
		for index := range output {
			output[index] = float32(int32(binary.LittleEndian.Uint32(source[index*4:])))
		}
	case dtype.I64:
		for index := range output {
			output[index] = float32(int64(binary.LittleEndian.Uint64(source[index*8:])))
		}
	case dtype.F64:
		for index := range output {
			value := math.Float64frombits(binary.LittleEndian.Uint64(source[index*8:]))
			output[index] = float32(value)
		}
	case dtype.Q8_0:
		for block := uint64(0); block < blocks; block++ {
			sourceOffset := block * traits.TypeSize
			outputOffset := block * traits.BlockSize
			scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
			for index := uint64(0); index < traits.BlockSize; index++ {
				quantized := int8(source[sourceOffset+2+index])
				output[outputOffset+index] = scale * float32(quantized)
			}
		}
	case dtype.Q8_1:
		for block := uint64(0); block < blocks; block++ {
			sourceOffset := block * traits.TypeSize
			outputOffset := block * traits.BlockSize
			scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
			for index := uint64(0); index < traits.BlockSize; index++ {
				quantized := int8(source[sourceOffset+4+index])
				output[outputOffset+index] = scale * float32(quantized)
			}
		}
	case dtype.Q8K:
		for block := uint64(0); block < blocks; block++ {
			sourceOffset := block * traits.TypeSize
			outputOffset := block * traits.BlockSize
			scale := math.Float32frombits(binary.LittleEndian.Uint32(source[sourceOffset:]))
			for index := uint64(0); index < traits.BlockSize; index++ {
				quantized := int8(source[sourceOffset+4+index])
				output[outputOffset+index] = scale * float32(quantized)
			}
		}
	case dtype.Q1_0:
		for block := uint64(0); block < blocks; block++ {
			sourceOffset := block * traits.TypeSize
			outputOffset := block * traits.BlockSize
			scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
			for index := uint64(0); index < traits.BlockSize; index++ {
				bit := (source[sourceOffset+2+index/8] >> (index % 8)) & 1
				value := -scale
				if bit != 0 {
					value = scale
				}
				output[outputOffset+index] = value
			}
		}
	case dtype.Q2_0:
		for block := uint64(0); block < blocks; block++ {
			sourceOffset := block * traits.TypeSize
			outputOffset := block * traits.BlockSize
			scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
			for index := uint64(0); index < traits.BlockSize; index++ {
				packed := source[sourceOffset+2+index/4]
				quantized := int((packed >> ((index % 4) * 2)) & 0x03)
				output[outputOffset+index] = float32(quantized-1) * scale
			}
		}
	case dtype.Q4_0, dtype.Q4_1:
		dequantizeQ4(dataType, source, output, blocks, traits)
	case dtype.Q5_0, dtype.Q5_1:
		dequantizeQ5(dataType, source, output, blocks, traits)
	case dtype.Q6K:
		dequantizeQ6K(source, output, blocks, traits)
	case dtype.Q2K:
		dequantizeQ2K(source, output, blocks, traits)
	case dtype.Q3K:
		dequantizeQ3K(source, output, blocks, traits)
	case dtype.Q4K:
		dequantizeQ4K(source, output, blocks, traits)
	case dtype.Q5K:
		dequantizeQ5K(source, output, blocks, traits)
	case dtype.IQ2XXS:
		dequantizeIQ2XXS(source, output, blocks, traits)
	case dtype.IQ2XS:
		dequantizeIQ2XS(source, output, blocks, traits)
	case dtype.IQ2S:
		dequantizeIQ2S(source, output, blocks, traits)
	case dtype.IQ3XXS:
		dequantizeIQ3XXS(source, output, blocks, traits)
	case dtype.IQ3S:
		dequantizeIQ3S(source, output, blocks, traits)
	case dtype.IQ1S:
		dequantizeIQ1S(source, output, blocks, traits)
	case dtype.IQ1M:
		dequantizeIQ1M(source, output, blocks, traits)
	case dtype.IQ4XS:
		dequantizeIQ4XS(source, output, blocks, traits)
	case dtype.IQ4NL:
		dequantizeIQ4NL(source, output, blocks, traits)
	case dtype.TQ2_0:
		dequantizeTQ2_0(source, output, blocks, traits)
	case dtype.TQ1_0:
		dequantizeTQ1_0(source, output, blocks, traits)
	case dtype.MXFP4:
		dequantizeMXFP4(source, output, blocks, traits)
	case dtype.NVFP4:
		dequantizeNVFP4(source, output, blocks, traits)
	default:
		return nil, fmt.Errorf("dequantization for %s is not implemented", dataType)
	}
	return output, nil
}

var iq4NLValues = [...]float32{
	-127, -104, -83, -65, -49, -35, -22, -10,
	1, 13, 25, 38, 53, 69, 89, 113,
}

func iqSignMask(index uint32) byte {
	index &= 0x7f
	parity := index
	parity ^= parity >> 4
	parity ^= parity >> 2
	parity ^= parity >> 1
	return byte(index | (parity&1)<<7)
}

func iqGrid64Lane(grid uint64, lane int) float32 {
	return float32(byte(grid >> uint(lane*8)))
}

func iqGrid32Lane(grid uint32, lane int) float32 {
	return float32(byte(grid >> uint(lane*8)))
}

func iqGrid1Lane(grid uint64, lane int) float32 {
	return float32(int8(byte(grid >> uint(lane*8))))
}

func dequantizeIQ2XXS(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		quantized := source[sourceOffset+2 : sourceOffset+66]
		for group := 0; group < 8; group++ {
			first := binary.LittleEndian.Uint32(quantized[group*8:])
			second := binary.LittleEndian.Uint32(quantized[group*8+4:])
			groupScale := scale * (0.5 + float32(second>>28)) * 0.25
			for subGroup := 0; subGroup < 4; subGroup++ {
				gridIndex := byte(first >> uint(subGroup*8))
				grid := iq2XXSGrid[gridIndex]
				signs := iqSignMask((second >> uint(subGroup*7)) & 0x7f)
				destination := outputOffset + uint64(group*32+subGroup*8)
				for lane := 0; lane < 8; lane++ {
					value := groupScale * iqGrid64Lane(grid, lane)
					if signs&(1<<uint(lane)) != 0 {
						value = -value
					}
					output[destination+uint64(lane)] = value
				}
			}
		}
	}
}

func dequantizeIQ2XS(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		quantized := source[sourceOffset+2 : sourceOffset+66]
		scales := source[sourceOffset+66 : sourceOffset+74]
		for group := 0; group < 8; group++ {
			packedScale := scales[group]
			groupScales := [2]float32{
				scale * (0.5 + float32(packedScale&0x0f)) * 0.25,
				scale * (0.5 + float32(packedScale>>4)) * 0.25,
			}
			for subGroup := 0; subGroup < 4; subGroup++ {
				packed := binary.LittleEndian.Uint16(
					quantized[(group*4+subGroup)*2:],
				)
				grid := iq2XSGrid[packed&0x01ff]
				signs := iqSignMask(uint32(packed >> 9))
				destination := outputOffset + uint64(group*32+subGroup*8)
				for lane := 0; lane < 8; lane++ {
					value := groupScales[subGroup/2] * iqGrid64Lane(grid, lane)
					if signs&(1<<uint(lane)) != 0 {
						value = -value
					}
					output[destination+uint64(lane)] = value
				}
			}
		}
	}
}

func dequantizeIQ2S(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		quantized := source[sourceOffset+2 : sourceOffset+66]
		high := source[sourceOffset+66 : sourceOffset+74]
		scales := source[sourceOffset+74 : sourceOffset+82]
		for group := 0; group < 8; group++ {
			packedScale := scales[group]
			groupScales := [2]float32{
				scale * (0.5 + float32(packedScale&0x0f)) * 0.25,
				scale * (0.5 + float32(packedScale>>4)) * 0.25,
			}
			for subGroup := 0; subGroup < 4; subGroup++ {
				gridIndex := uint16(quantized[group*4+subGroup]) |
					(uint16(high[group])<<uint(8-2*subGroup))&0x0300
				grid := iq2SGrid[gridIndex]
				signs := quantized[32+group*4+subGroup]
				destination := outputOffset + uint64(group*32+subGroup*8)
				for lane := 0; lane < 8; lane++ {
					value := groupScales[subGroup/2] * iqGrid64Lane(grid, lane)
					if signs&(1<<uint(lane)) != 0 {
						value = -value
					}
					output[destination+uint64(lane)] = value
				}
			}
		}
	}
}

func dequantizeIQ3XXS(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		quantized := source[sourceOffset+2 : sourceOffset+66]
		scalesAndSigns := source[sourceOffset+66 : sourceOffset+98]
		for group := 0; group < 8; group++ {
			packed := binary.LittleEndian.Uint32(scalesAndSigns[group*4:])
			groupScale := scale * (0.5 + float32(packed>>28)) * 0.5
			for subGroup := 0; subGroup < 4; subGroup++ {
				first := iq3XXSGrid[quantized[group*8+subGroup*2]]
				second := iq3XXSGrid[quantized[group*8+subGroup*2+1]]
				signs := iqSignMask((packed >> uint(subGroup*7)) & 0x7f)
				destination := outputOffset + uint64(group*32+subGroup*8)
				for lane := 0; lane < 4; lane++ {
					value := groupScale * iqGrid32Lane(first, lane)
					if signs&(1<<uint(lane)) != 0 {
						value = -value
					}
					output[destination+uint64(lane)] = value
					value = groupScale * iqGrid32Lane(second, lane)
					if signs&(1<<uint(lane+4)) != 0 {
						value = -value
					}
					output[destination+uint64(lane+4)] = value
				}
			}
		}
	}
}

func dequantizeIQ3S(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		quantized := source[sourceOffset+2 : sourceOffset+66]
		high := source[sourceOffset+66 : sourceOffset+74]
		signs := source[sourceOffset+74 : sourceOffset+106]
		scales := source[sourceOffset+106 : sourceOffset+110]
		for pair := 0; pair < 4; pair++ {
			packedScale := scales[pair]
			groupScales := [2]float32{
				scale * float32(1+2*int(packedScale&0x0f)),
				scale * float32(1+2*int(packedScale>>4)),
			}
			for half := 0; half < 2; half++ {
				qBase := pair*16 + half*8
				signBase := pair*8 + half*4
				highBits := high[pair*2+half]
				for subGroup := 0; subGroup < 4; subGroup++ {
					firstIndex := uint16(quantized[qBase+subGroup*2]) |
						(uint16(highBits)<<uint(8-2*subGroup))&0x0100
					secondIndex := uint16(quantized[qBase+subGroup*2+1]) |
						(uint16(highBits)<<uint(7-2*subGroup))&0x0100
					first := iq3SGrid[firstIndex]
					second := iq3SGrid[secondIndex]
					signMask := signs[signBase+subGroup]
					destination := outputOffset +
						uint64(pair*64+half*32+subGroup*8)
					for lane := 0; lane < 4; lane++ {
						value := groupScales[half] * iqGrid32Lane(first, lane)
						if signMask&(1<<uint(lane)) != 0 {
							value = -value
						}
						output[destination+uint64(lane)] = value
						value = groupScales[half] * iqGrid32Lane(second, lane)
						if signMask&(1<<uint(lane+4)) != 0 {
							value = -value
						}
						output[destination+uint64(lane+4)] = value
					}
				}
			}
		}
	}
}

func dequantizeIQ1S(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	const deltaMagnitude = float32(0.125)
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		quantized := source[sourceOffset+2 : sourceOffset+34]
		high := source[sourceOffset+34 : sourceOffset+50]
		for group := 0; group < 8; group++ {
			packedHigh := binary.LittleEndian.Uint16(high[group*2:])
			groupScale := scale * float32(2*((packedHigh>>12)&7)+1)
			delta := deltaMagnitude
			if packedHigh&0x8000 != 0 {
				delta = -delta
			}
			for subGroup := 0; subGroup < 4; subGroup++ {
				gridIndex := uint16(quantized[group*4+subGroup]) |
					((packedHigh>>uint(subGroup*3))&7)<<8
				grid := iq1SGrid[gridIndex]
				destination := outputOffset + uint64(group*32+subGroup*8)
				for lane := 0; lane < 8; lane++ {
					output[destination+uint64(lane)] =
						groupScale * (iqGrid1Lane(grid, lane) + delta)
				}
			}
		}
	}
}

func dequantizeIQ1M(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	const deltaMagnitude = float32(0.125)
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		quantized := source[sourceOffset : sourceOffset+32]
		high := source[sourceOffset+32 : sourceOffset+48]
		scales := source[sourceOffset+48 : sourceOffset+56]
		scaleWords := [4]uint16{
			binary.LittleEndian.Uint16(scales[0:]),
			binary.LittleEndian.Uint16(scales[2:]),
			binary.LittleEndian.Uint16(scales[4:]),
			binary.LittleEndian.Uint16(scales[6:]),
		}
		scaleBits := (scaleWords[0] >> 12) |
			((scaleWords[1] >> 8) & 0x00f0) |
			((scaleWords[2] >> 4) & 0x0f00) |
			(scaleWords[3] & 0xf000)
		scale := Float16ToFloat32(scaleBits)
		for group := 0; group < 8; group++ {
			scaleWord := scaleWords[group/2]
			scaleShift := uint(6 * (group % 2))
			groupScales := [2]float32{
				scale * float32(2*((scaleWord>>scaleShift)&7)+1),
				scale * float32(2*((scaleWord>>(scaleShift+3))&7)+1),
			}
			qBase := group * 4
			highBase := group * 2
			gridIndices := [4]uint16{
				uint16(quantized[qBase]) |
					(uint16(high[highBase])<<8)&0x0700,
				uint16(quantized[qBase+1]) |
					(uint16(high[highBase])<<4)&0x0700,
				uint16(quantized[qBase+2]) |
					(uint16(high[highBase+1])<<8)&0x0700,
				uint16(quantized[qBase+3]) |
					(uint16(high[highBase+1])<<4)&0x0700,
			}
			deltas := [4]float32{
				deltaMagnitude,
				deltaMagnitude,
				deltaMagnitude,
				deltaMagnitude,
			}
			if high[highBase]&0x08 != 0 {
				deltas[0] = -deltaMagnitude
			}
			if high[highBase]&0x80 != 0 {
				deltas[1] = -deltaMagnitude
			}
			if high[highBase+1]&0x08 != 0 {
				deltas[2] = -deltaMagnitude
			}
			if high[highBase+1]&0x80 != 0 {
				deltas[3] = -deltaMagnitude
			}
			for subGroup := 0; subGroup < 4; subGroup++ {
				grid := iq1SGrid[gridIndices[subGroup]]
				destination := outputOffset + uint64(group*32+subGroup*8)
				for lane := 0; lane < 8; lane++ {
					output[destination+uint64(lane)] =
						groupScales[subGroup/2] *
							(iqGrid1Lane(grid, lane) + deltas[subGroup])
				}
			}
		}
	}
}

func dequantizeIQ4NL(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		quantized := source[sourceOffset+2 : sourceOffset+18]
		for lane, packed := range quantized {
			output[outputOffset+uint64(lane)] =
				scale * iq4NLValues[packed&0x0f]
			output[outputOffset+uint64(lane+16)] =
				scale * iq4NLValues[packed>>4]
		}
	}
}

func dequantizeTQ2_0(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		quantized := source[sourceOffset : sourceOffset+64]
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+64:]))
		for half := 0; half < 2; half++ {
			for pair := 0; pair < 4; pair++ {
				destination := outputOffset + uint64(half*128+pair*32)
				for lane := 0; lane < 32; lane++ {
					value := (quantized[half*32+lane] >> uint(pair*2)) & 0x03
					output[destination+uint64(lane)] =
						float32(int(value)-1) * scale
				}
			}
		}
	}
}

func dequantizeTQ1_0(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	powersOfThree := [...]byte{1, 3, 9, 27, 81}
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		quantized := source[sourceOffset : sourceOffset+48]
		high := source[sourceOffset+48 : sourceOffset+52]
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+52:]))
		destination := outputOffset
		for _, width := range []int{32, 16} {
			sourceBase := 0
			if width == 16 {
				sourceBase = 32
			}
			for trit := 0; trit < 5; trit++ {
				for lane := 0; lane < width; lane++ {
					packed := byte(uint16(quantized[sourceBase+lane]) *
						uint16(powersOfThree[trit]))
					value := (uint16(packed) * 3) >> 8
					output[destination] = float32(int(value)-1) * scale
					destination++
				}
			}
		}
		for trit := 0; trit < 4; trit++ {
			for lane := 0; lane < 4; lane++ {
				packed := byte(uint16(high[lane]) *
					uint16(powersOfThree[trit]))
				value := (uint16(packed) * 3) >> 8
				output[destination] = float32(int(value)-1) * scale
				destination++
			}
		}
	}
}

var mxfp4Values = [...]float32{
	0, 1, 2, 3, 4, 6, 8, 12,
	0, -1, -2, -3, -4, -6, -8, -12,
}

func dequantizeMXFP4(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		exponent := source[sourceOffset]
		var scale float32
		if exponent < 2 {
			scale = math.Float32frombits(0x00200000 << exponent)
		} else {
			scale = math.Float32frombits(uint32(exponent-1) << 23)
		}
		quantized := source[sourceOffset+1 : sourceOffset+17]
		for lane, packed := range quantized {
			output[outputOffset+uint64(lane)] =
				scale * mxfp4Values[packed&0x0f]
			output[outputOffset+uint64(lane+16)] =
				scale * mxfp4Values[packed>>4]
		}
	}
}

func ue4m3ToFloat32(value byte) float32 {
	if value == 0 || value == 0x7f {
		return 0
	}
	exponent := (value >> 3) & 0x0f
	mantissa := value & 0x07
	if exponent == 0 {
		return float32(mantissa) / 1024
	}
	exponentScale := math.Float32frombits(uint32(119+exponent) << 23)
	return (1 + float32(mantissa)/8) * exponentScale
}

func dequantizeNVFP4(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scales := source[sourceOffset : sourceOffset+4]
		quantized := source[sourceOffset+4 : sourceOffset+36]
		for subBlock := 0; subBlock < 4; subBlock++ {
			scale := ue4m3ToFloat32(scales[subBlock])
			sourceBase := subBlock * 8
			destination := outputOffset + uint64(subBlock*16)
			for lane := 0; lane < 8; lane++ {
				packed := quantized[sourceBase+lane]
				output[destination+uint64(lane)] =
					scale * mxfp4Values[packed&0x0f]
				output[destination+uint64(lane+8)] =
					scale * mxfp4Values[packed>>4]
			}
		}
	}
}

func dequantizeIQ4XS(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		highScales := binary.LittleEndian.Uint16(source[sourceOffset+2:])
		lowScales := source[sourceOffset+4 : sourceOffset+8]
		quantized := source[sourceOffset+8 : sourceOffset+136]
		for group := 0; group < 8; group++ {
			packedScale := (lowScales[group/2] >> uint(4*(group%2))) & 0x0f
			packedScale |= byte((highScales>>uint(2*group))&0x03) << 4
			groupScale := scale * float32(int(packedScale)-32)
			groupOffset := outputOffset + uint64(group*32)
			quantizedOffset := group * 16
			for lane := 0; lane < 16; lane++ {
				packed := quantized[quantizedOffset+lane]
				output[groupOffset+uint64(lane)] =
					groupScale * iq4NLValues[packed&0x0f]
				output[groupOffset+uint64(lane+16)] =
					groupScale * iq4NLValues[packed>>4]
			}
		}
	}
}

func dequantizeQ2K(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scales := source[sourceOffset : sourceOffset+16]
		quantized := source[sourceOffset+16 : sourceOffset+80]
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+80:]))
		minimum := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+82:]))
		for group := 0; group < 16; group++ {
			scaleAndMin := scales[group]
			delta := scale * float32(scaleAndMin&0x0f)
			minimumValue := minimum * float32(scaleAndMin>>4)
			half := group / 8
			groupInHalf := group % 8
			shift := uint((groupInHalf / 2) * 2)
			quantBase := half*32 + (groupInHalf%2)*16
			destination := outputOffset + uint64(group*16)
			for lane := 0; lane < 16; lane++ {
				value := (quantized[quantBase+lane] >> shift) & 0x03
				output[destination+uint64(lane)] =
					delta*float32(value) - minimumValue
			}
		}
	}
}

func dequantizeQ3K(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		highMasks := source[sourceOffset : sourceOffset+32]
		quantized := source[sourceOffset+32 : sourceOffset+96]
		scales := source[sourceOffset+96 : sourceOffset+108]
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+108:]))
		for group := 0; group < 16; group++ {
			lowScale := scales[group%8]
			if group >= 8 {
				lowScale >>= 4
			} else {
				lowScale &= 0x0f
			}
			highScale := (scales[8+group%4] >> uint(2*(group/4))) & 0x03
			groupScale := int(lowScale|highScale<<4) - 32
			half := group / 8
			groupInHalf := group % 8
			shift := uint((groupInHalf / 2) * 2)
			mask := byte(1 << uint(group/2))
			quantBase := half*32 + (groupInHalf%2)*16
			maskBase := (groupInHalf % 2) * 16
			destination := outputOffset + uint64(group*16)
			for lane := 0; lane < 16; lane++ {
				value := int((quantized[quantBase+lane] >> shift) & 0x03)
				if highMasks[maskBase+lane]&mask == 0 {
					value -= 4
				}
				output[destination+uint64(lane)] =
					scale * float32(groupScale) * float32(value)
			}
		}
	}
}

func scaleMinK4(index int, packed []byte) (scale, minimum byte) {
	if index < 4 {
		return packed[index] & 0x3f, packed[index+4] & 0x3f
	}
	scale = (packed[index+4] & 0x0f) | (packed[index-4]>>6)<<4
	minimum = (packed[index+4] >> 4) | (packed[index]>>6)<<4
	return scale, minimum
}

func dequantizeQ4K(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		minimum := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+2:]))
		scales := source[sourceOffset+4 : sourceOffset+16]
		quantized := source[sourceOffset+16 : sourceOffset+144]
		for group := 0; group < 8; group++ {
			groupScale, groupMinimum := scaleMinK4(group, scales)
			delta := scale * float32(groupScale)
			minimumValue := minimum * float32(groupMinimum)
			packedBase := (group / 2) * 32
			destination := outputOffset + uint64(group*32)
			for lane := 0; lane < 32; lane++ {
				value := quantized[packedBase+lane]
				if group%2 == 0 {
					value &= 0x0f
				} else {
					value >>= 4
				}
				output[destination+uint64(lane)] =
					delta*float32(value) - minimumValue
			}
		}
	}
}

func dequantizeQ5K(source []byte, output []float32, blocks uint64, traits dtype.Traits) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		minimum := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+2:]))
		scales := source[sourceOffset+4 : sourceOffset+16]
		highBits := source[sourceOffset+16 : sourceOffset+48]
		quantized := source[sourceOffset+48 : sourceOffset+176]
		for group := 0; group < 8; group++ {
			groupScale, groupMinimum := scaleMinK4(group, scales)
			delta := scale * float32(groupScale)
			minimumValue := minimum * float32(groupMinimum)
			packedBase := (group / 2) * 32
			highMask := byte(1 << uint(group))
			destination := outputOffset + uint64(group*32)
			for lane := 0; lane < 32; lane++ {
				value := quantized[packedBase+lane]
				if group%2 == 0 {
					value &= 0x0f
				} else {
					value >>= 4
				}
				if highBits[lane]&highMask != 0 {
					value += 16
				}
				output[destination+uint64(lane)] =
					delta*float32(value) - minimumValue
			}
		}
	}
}

func dequantizeQ6K(
	source []byte,
	output []float32,
	blocks uint64,
	traits dtype.Traits,
) {
	const (
		lowerBytes = 128
		highBytes  = 64
		scaleBytes = 16
	)
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		lower := source[sourceOffset : sourceOffset+lowerBytes]
		high := source[sourceOffset+lowerBytes : sourceOffset+lowerBytes+highBytes]
		scales := source[sourceOffset+lowerBytes+highBytes : sourceOffset+lowerBytes+highBytes+scaleBytes]
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(
			source[sourceOffset+lowerBytes+highBytes+scaleBytes:],
		))
		for group := 0; group < 2; group++ {
			lowerBase := group * 64
			highBase := group * 32
			scaleBase := group * 8
			destination := outputOffset + uint64(group*128)
			for column := 0; column < 32; column++ {
				scaleIndex := column / 16
				q1 := int(lower[lowerBase+column]&0x0f) |
					int((high[highBase+column]>>0)&0x03)<<4
				q2 := int(lower[lowerBase+column+32]&0x0f) |
					int((high[highBase+column]>>2)&0x03)<<4
				q3 := int(lower[lowerBase+column]>>4) |
					int((high[highBase+column]>>4)&0x03)<<4
				q4 := int(lower[lowerBase+column+32]>>4) |
					int((high[highBase+column]>>6)&0x03)<<4
				output[destination+uint64(column)] =
					scale * float32(int8(scales[scaleBase+scaleIndex])) * float32(q1-32)
				output[destination+uint64(column+32)] =
					scale * float32(int8(scales[scaleBase+scaleIndex+2])) * float32(q2-32)
				output[destination+uint64(column+64)] =
					scale * float32(int8(scales[scaleBase+scaleIndex+4])) * float32(q3-32)
				output[destination+uint64(column+96)] =
					scale * float32(int8(scales[scaleBase+scaleIndex+6])) * float32(q4-32)
			}
		}
	}
}

func dequantizeQ4(
	dataType dtype.Type,
	source []byte,
	output []float32,
	blocks uint64,
	traits dtype.Traits,
) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		minimum := float32(0)
		quantOffset := sourceOffset + 2
		if dataType == dtype.Q4_1 {
			minimum = Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+2:]))
			quantOffset += 2
		}
		for index := uint64(0); index < traits.BlockSize/2; index++ {
			packed := source[quantOffset+index]
			low := int(packed & 0x0f)
			high := int(packed >> 4)
			if dataType == dtype.Q4_0 {
				low -= 8
				high -= 8
			}
			output[outputOffset+index] = float32(low)*scale + minimum
			output[outputOffset+index+traits.BlockSize/2] = float32(high)*scale + minimum
		}
	}
}

func dequantizeQ5(
	dataType dtype.Type,
	source []byte,
	output []float32,
	blocks uint64,
	traits dtype.Traits,
) {
	for block := uint64(0); block < blocks; block++ {
		sourceOffset := block * traits.TypeSize
		outputOffset := block * traits.BlockSize
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset:]))
		minimum := float32(0)
		highOffset := sourceOffset + 2
		if dataType == dtype.Q5_1 {
			minimum = Float16ToFloat32(binary.LittleEndian.Uint16(source[sourceOffset+2:]))
			highOffset += 2
		}
		highBits := binary.LittleEndian.Uint32(source[highOffset:])
		quantOffset := highOffset + 4
		for index := uint64(0); index < traits.BlockSize/2; index++ {
			packed := source[quantOffset+index]
			low := int(packed&0x0f) | int((highBits>>index)&1)<<4
			high := int(packed>>4) | int((highBits>>(index+16))&1)<<4
			if dataType == dtype.Q5_0 {
				low -= 16
				high -= 16
			}
			output[outputOffset+index] = float32(low)*scale + minimum
			output[outputOffset+index+traits.BlockSize/2] = float32(high)*scale + minimum
		}
	}
}

// Float16ToFloat32: converts IEEE 754 binary16 bit pattern
func Float16ToFloat32(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exponent := uint32(value>>10) & 0x1f
	mantissa := uint32(value & 0x03ff)

	var bits uint32
	switch exponent {
	case 0:
		if mantissa == 0 {
			bits = sign
			break
		}
		exponent = 113
		for mantissa&0x0400 == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= 0x03ff
		bits = sign | exponent<<23 | mantissa<<13
	case 0x1f:
		bits = sign | 0x7f800000 | mantissa<<13
	default:
		bits = sign | (exponent+112)<<23 | mantissa<<13
	}
	return math.Float32frombits(bits)
}
