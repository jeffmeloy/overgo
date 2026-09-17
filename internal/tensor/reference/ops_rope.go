package reference

import (
	"cmp"
	"errors"
	"math"
	"slices"

	"overgo/internal/tensor"
)

func ropeNeoX(shape tensor.Shape, inputs []Value, attributes tensor.RoPENeoXAttributes) (Value, error) {
	input := inputs[0]
	width := int(shape.Dims[0])
	heads := int(shape.Dims[1])
	tokens := int(shape.Dims[2])
	batches := int(shape.Dims[3])
	rotary := int(attributes.RotaryDimensions)
	if rotary <= 0 || rotary%2 != 0 || rotary > width {
		return Value{}, errors.New("invalid rope_neox rotary width")
	}
	if len(attributes.Positions) != tokens {
		return Value{}, errors.New("rope_neox position count differs from token count")
	}
	factors, err := ropeFrequencyFactors(inputs, rotary)
	if err != nil {
		return Value{}, err
	}
	output := slices.Clone(input.Data)
	half := rotary / 2
	for batch := range batches {
		for tokenIndex, position := range attributes.Positions {
			for head := range heads {
				offset := ((batch*tokens+tokenIndex)*heads + head) * width
				for pairIndex := range half {
					cosine, sine := ropeCosSin(attributes, pairIndex, rotary, position, factors[pairIndex])
					x0 := input.Data[offset+pairIndex]
					x1 := input.Data[offset+pairIndex+half]
					output[offset+pairIndex] = x0*cosine - x1*sine
					output[offset+pairIndex+half] = x0*sine + x1*cosine
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func ropeNormal(shape tensor.Shape, inputs []Value, attributes tensor.RoPEAttributes) (Value, error) {
	input := inputs[0]
	width := int(shape.Dims[0])
	heads := int(shape.Dims[1])
	tokens := int(shape.Dims[2])
	batches := int(shape.Dims[3])
	rotary := int(attributes.RotaryDimensions)
	if rotary <= 0 || rotary%2 != 0 || rotary > width {
		return Value{}, errors.New("invalid rope_normal rotary width")
	}
	if len(attributes.Positions) != tokens {
		return Value{}, errors.New("rope_normal position count differs from token count")
	}
	factors, err := ropeFrequencyFactors(inputs, rotary)
	if err != nil {
		return Value{}, err
	}
	output := slices.Clone(input.Data)
	for batch := range batches {
		for tokenIndex, position := range attributes.Positions {
			for head := range heads {
				offset := ((batch*tokens+tokenIndex)*heads + head) * width
				for pairIndex := range rotary / 2 {
					cosine, sine := ropeCosSin(attributes, pairIndex, rotary, position, factors[pairIndex])
					first := offset + pairIndex*2
					x0 := input.Data[first]
					x1 := input.Data[first+1]
					output[first] = x0*cosine - x1*sine
					output[first+1] = x0*sine + x1*cosine
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func ropeCosSin(
	attributes tensor.RoPEAttributes,
	pairIndex, rotary int,
	position uint32,
	factor float32,
) (float32, float32) {
	thetaExtrapolated := float64(position) * math.Pow(
		float64(attributes.FrequencyBase), -2*float64(pairIndex)/float64(rotary),
	) / float64(factor)
	theta := float64(attributes.FrequencyScale) * thetaExtrapolated
	magnitude := float64(attributes.AttentionFactor)
	magnitude = cmp.Or(magnitude, 1)
	if attributes.ExtFactor != 0 && attributes.OriginalContext > 0 {
		correction := func(rotations float32) float64 {
			return float64(rotary) * math.Log(
				float64(attributes.OriginalContext)/(float64(rotations)*2*math.Pi),
			) / (2 * math.Log(float64(attributes.FrequencyBase)))
		}
		low := math.Floor(correction(attributes.BetaFast))
		high := math.Ceil(correction(attributes.BetaSlow))
		low = max(0, min(float64(rotary-1), low))
		high = max(0, min(float64(rotary-1), high))
		ramp := 1 - min(1, max(0, (float64(pairIndex)-low)/max(0.001, high-low)))
		mix := ramp * float64(attributes.ExtFactor)
		theta = theta*(1-mix) + thetaExtrapolated*mix
		magnitude *= 1 + 0.1*math.Log(1/float64(attributes.FrequencyScale))
	}
	return float32(math.Cos(theta) * magnitude), float32(math.Sin(theta) * magnitude)
}

func ropeFrequencyFactors(inputs []Value, rotary int) ([]float32, error) {
	factors := make([]float32, rotary/2)
	for index := range factors {
		factors[index] = 1
	}
	if len(inputs) == 1 {
		return factors, nil
	}
	if len(inputs) != 2 ||
		inputs[1].Shape.Rank != 1 ||
		inputs[1].Shape.Dims[0] != uint64(len(factors)) ||
		len(inputs[1].Data) != len(factors) {
		return nil, errors.New("RoPE frequency factors have invalid shape")
	}
	for index, factor := range inputs[1].Data {
		if factor <= 0 || math.IsNaN(float64(factor)) || math.IsInf(float64(factor), 0) {
			return nil, errors.New("RoPE frequency factor must be finite and positive")
		}
		factors[index] = factor
	}
	return factors, nil
}

func ropeMulti(
	shape tensor.Shape,
	input Value,
	attributes tensor.RoPEMultiAttributes,
) (Value, error) {
	width := int(shape.Dims[0])
	heads := int(shape.Dims[1])
	tokens := int(shape.Dims[2])
	batches := int(shape.Dims[3])
	rotary := int(attributes.RotaryDimensions)
	half := rotary / 2
	var sectionPairs int
	for _, section := range attributes.Sections {
		sectionPairs += int(section)
	}
	if rotary <= 0 || rotary%2 != 0 || rotary > width ||
		sectionPairs <= 0 {
		return Value{}, errors.New("invalid rope_multi dimensions")
	}
	output := slices.Clone(input.Data)
	for batch := range batches {
		for token := range tokens {
			for head := range heads {
				offset := ((batch*tokens+token)*heads + head) * width
				for pair := range half {
					sector := pair % sectionPairs
					axis := 0
					boundary := int(attributes.Sections[0])
					for axis < 3 && sector >= boundary {
						axis++
						boundary += int(attributes.Sections[axis])
					}
					if attributes.InterleavedSections {
						// IMRoPE cycles temporal, height and width; exhausted sections use the extra axis.
						const spatialAxes = tensor.MaxDimensions - 1
						axis = sector % spatialAxes
						if sector >= spatialAxes*int(attributes.Sections[axis]) {
							axis = spatialAxes
						}
					}
					theta := float64(attributes.Positions[axis][token]) * float64(attributes.FrequencyScale) * math.Pow(
						float64(attributes.FrequencyBase),
						-2*float64(pair)/float64(rotary),
					)
					cosine := float32(math.Cos(theta))
					sine := float32(math.Sin(theta))
					first := offset + pair*2
					x0 := input.Data[first]
					x1 := input.Data[first+1]
					output[first] = x0*cosine - x1*sine
					output[first+1] = x0*sine + x1*cosine
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func repeatHeads(shape tensor.Shape, input Value) Value {
	width := int(input.Shape.Dims[0])
	tokens := int(input.Shape.Dims[2])
	heads := int(shape.Dims[1])
	output := make([]float32, width*heads*tokens)
	for token := range tokens {
		source := input.Data[token*width : (token+1)*width]
		for head := range heads {
			destination := (token*heads + head) * width
			copy(output[destination:destination+width], source)
		}
	}
	return Value{Shape: shape, Data: output}
}
