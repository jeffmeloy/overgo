package representation

import (
	"errors"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// TwoLayerProjectionF32 is a validated input-to-hidden-to-output projection
// whose hidden and output widths are equal.
type TwoLayerProjectionF32 struct {
	InputWidth, OutputWidth  int
	InputWeight, InputBias   []float32
	OutputWeight, OutputBias []float32
}

// CompileTwoLayerProjectionF32 derives projection geometry from flat weights.
func CompileTwoLayerProjectionF32(inputWidth int, inputWeight, inputBias, outputWeight, outputBias []float32) (TwoLayerProjectionF32, error) {
	outputWidth, err := checked.Rows(inputWeight, inputWidth)
	if err != nil {
		return TwoLayerProjectionF32{}, errors.New("representation: invalid projection input weights")
	}
	if err := checked.Length(inputBias, outputWidth); err != nil {
		return TwoLayerProjectionF32{}, errors.New("representation: invalid projection input bias")
	}
	if err := checked.Length(outputWeight, outputWidth, outputWidth); err != nil {
		return TwoLayerProjectionF32{}, errors.New("representation: invalid projection output weights")
	}
	if err := checked.Length(outputBias, outputWidth); err != nil {
		return TwoLayerProjectionF32{}, errors.New("representation: invalid projection output bias")
	}
	return TwoLayerProjectionF32{
		InputWidth: inputWidth, OutputWidth: outputWidth,
		InputWeight: inputWeight, InputBias: inputBias,
		OutputWeight: outputWeight, OutputBias: outputBias,
	}, nil
}

// ProjectPaddedF32 projects active sequence rows through Linear-GELU-Linear,
// retains one projected zero-input row, and broadcasts it over remaining rows.
func ProjectPaddedF32(context []float32, contextRows, capacity int, projection TwoLayerProjectionF32, bf16 bool) ([]float32, error) {
	if err := checked.Length(context, contextRows, projection.InputWidth); err != nil {
		return nil, errors.New("representation: invalid projection context")
	}
	activeRows, err := ActivePrefixRows(contextRows, capacity)
	if err != nil {
		return nil, err
	}
	inputElements, ok := checked.MulInt(activeRows, projection.InputWidth)
	if !ok {
		return nil, errors.New("representation: projection input overflows")
	}
	input := make([]float32, inputElements)
	copy(input, context)
	hidden := hostmath.LinearF64BiasFirstNew(input, projection.InputWeight, projection.InputBias, activeRows, projection.InputWidth, projection.OutputWidth)
	if bf16 {
		dtype.RoundBF16Slice(hidden)
	}
	hostmath.GELUTanhInPlace(hidden)
	if bf16 {
		dtype.RoundBF16Slice(hidden)
	}
	projected := hostmath.LinearF64New(hidden, projection.OutputWeight, projection.OutputBias, activeRows, projection.OutputWidth, projection.OutputWidth)
	if bf16 {
		dtype.RoundBF16Slice(projected)
	}
	outputElements, ok := checked.MulInt(capacity, projection.OutputWidth)
	if !ok {
		return nil, errors.New("representation: projection output overflows")
	}
	output := make([]float32, outputElements)
	activeElements, _ := checked.MulInt(contextRows, projection.OutputWidth)
	copy(output, projected[:activeElements])
	if contextRows < capacity {
		padding := projected[activeElements : activeElements+projection.OutputWidth]
		for row := contextRows; row < capacity; row++ {
			copy(output[row*projection.OutputWidth:], padding)
		}
	}
	return output, nil
}
