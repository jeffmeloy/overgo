package hostmath

import (
	"errors"

	"overgo/internal/checked"
)

// NormalizeProjectRowsToChannelsF64 applies a per-input affine followed by a
// bias-free linear projection, writing channel-major output with f64 sums.
func NormalizeProjectRowsToChannelsF64(out, input, scale, offset, weight []float32, rows, inputWidth, outputWidth int) error {
	if err := checked.Length(input, rows, inputWidth); err != nil {
		return errors.New("hostmath: invalid normalized projection input")
	}
	if err := checked.Length(scale, inputWidth); err != nil {
		return errors.New("hostmath: invalid normalized projection scale")
	}
	if err := checked.Length(offset, inputWidth); err != nil {
		return errors.New("hostmath: invalid normalized projection offset")
	}
	if err := checked.Length(weight, outputWidth, inputWidth); err != nil {
		return errors.New("hostmath: invalid normalized projection weights")
	}
	if err := checked.Length(out, outputWidth, rows); err != nil {
		return errors.New("hostmath: invalid normalized projection output")
	}
	for row := range rows {
		inputRow := input[row*inputWidth : (row+1)*inputWidth]
		for output := range outputWidth {
			weightRow := weight[output*inputWidth : (output+1)*inputWidth]
			var sum float64
			for inputIndex := range inputWidth {
				normalized := inputRow[inputIndex]*scale[inputIndex] + offset[inputIndex]
				sum += float64(normalized) * float64(weightRow[inputIndex])
			}
			out[output*rows+row] = float32(sum)
		}
	}
	return nil
}
