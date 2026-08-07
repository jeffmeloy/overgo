package projector

import (
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/tensor"
)

const (
	maxPixelMergeRows           uint64 = 1 << 32
	defaultVisionMaxAspectRatio        = 200
	temporalPatchChannels              = 3
	temporalPatchFrames                = 2
	rgbChannelCount                    = 3
	attentionProjectionCount           = 3
	rgba16To8Shift                     = 8
	maxUint8Channel                    = 255
	opaqueAlpha                        = 255
)

func normalizedImageChannel(value uint32) float32 {
	return float32(value>>rgba16To8Shift) / maxUint8Channel
}

type pixelMergePlan struct {
	inputRows  int
	outputRows int
	indexSets  [][]uint32
}

func newPixelMergePlan(height, width, merge int) (pixelMergePlan, error) {
	if height <= 0 || width <= 0 || merge <= 0 || height%merge != 0 || width%merge != 0 {
		return pixelMergePlan{}, errors.New("projector: invalid pixel merge geometry")
	}
	inputElements, ok := checked.Mul64(uint64(height), uint64(width))
	if !ok {
		return pixelMergePlan{}, errors.New("projector: pixel merge input size overflow")
	}
	inputRows, ok := checked.Int(inputElements)
	if !ok || inputElements > maxPixelMergeRows {
		return pixelMergePlan{}, errors.New("projector: pixel merge input size overflow")
	}
	mergeElements, ok := checked.Mul64(uint64(merge), uint64(merge))
	if !ok {
		return pixelMergePlan{}, errors.New("projector: pixel merge size overflow")
	}
	indexCount, ok := checked.Int(mergeElements)
	if !ok {
		return pixelMergePlan{}, errors.New("projector: pixel merge size overflow")
	}
	outputRows, ok := checked.Int(inputElements / mergeElements)
	if !ok {
		return pixelMergePlan{}, errors.New("projector: pixel merge output size overflow")
	}
	indexSets := make([][]uint32, indexCount)
	for index := range indexSets {
		indexSets[index] = make([]uint32, 0, outputRows)
	}
	for blockY := 0; blockY < height/merge; blockY++ {
		for blockX := 0; blockX < width/merge; blockX++ {
			for y := 0; y < merge; y++ {
				for x := 0; x < merge; x++ {
					offset := y*merge + x
					indexSets[offset] = append(indexSets[offset], uint32((blockY*merge+y)*width+blockX*merge+x))
				}
			}
		}
	}
	return pixelMergePlan{inputRows: inputRows, outputRows: outputRows, indexSets: indexSets}, nil
}

func (plan pixelMergePlan) graph(builder *tensor.Builder, input *tensor.Tensor) *tensor.Tensor {
	output := builder.GetRows(input, plan.indexSets[0])
	for offset := 1; offset < len(plan.indexSets); offset++ {
		output = builder.Concat(output, builder.GetRows(input, plan.indexSets[offset]), 0)
	}
	return output
}

func (plan pixelMergePlan) shuffle(values []float32, width int) ([]float32, error) {
	if width <= 0 {
		return nil, fmt.Errorf("projector: pixel merge input has %d values for %d rows at width %d", len(values), plan.inputRows, width)
	}
	inputElements, ok := checked.Mul64(uint64(plan.inputRows), uint64(width))
	expected, ok := checked.Int(inputElements)
	if !ok || len(values) != expected {
		return nil, fmt.Errorf("projector: pixel merge input has %d values for %d rows at width %d", len(values), plan.inputRows, width)
	}
	outputElements, ok := checked.Mul64(uint64(plan.outputRows), uint64(width))
	if ok {
		outputElements, ok = checked.Mul64(outputElements, uint64(len(plan.indexSets)))
	}
	outputSize, ok := checked.Int(outputElements)
	if !ok {
		return nil, errors.New("projector: pixel merge output size overflow")
	}
	output := make([]float32, outputSize)
	for offset, rows := range plan.indexSets {
		for outputRow, inputRow := range rows {
			destination := (outputRow*len(plan.indexSets) + offset) * width
			source := int(inputRow) * width
			copy(output[destination:destination+width], values[source:source+width])
		}
	}
	return output, nil
}

func splitTemporalPatchPairs(
	values []float32,
	rows, patchArea int,
) ([]float32, []float32, int, error) {
	if rows <= 0 || patchArea <= 0 {
		return nil, nil, 0, errors.New("projector: invalid temporal patch geometry")
	}
	temporalElements, ok := checked.Mul64(uint64(rows), uint64(patchArea))
	if !ok {
		return nil, nil, 0, errors.New("projector: temporal patch size overflow")
	}
	temporalElements, ok = checked.Mul64(temporalElements, temporalPatchChannels*temporalPatchFrames)
	if !ok {
		return nil, nil, 0, errors.New("projector: temporal patch size overflow")
	}
	expected, ok := checked.Int(temporalElements)
	if !ok {
		return nil, nil, 0, errors.New("projector: temporal patch size overflow")
	}
	if len(values) != expected {
		return nil, nil, 0, fmt.Errorf("projector: temporal patch tensor has %d values, want %d", len(values), expected)
	}
	temporalWidth := temporalPatchChannels * patchArea
	first := make([]float32, rows*temporalWidth)
	second := make([]float32, rows*temporalWidth)
	for row := range rows {
		source := values[row*temporalWidth*temporalPatchFrames:]
		for color := range temporalPatchChannels {
			destination := row*temporalWidth + color*patchArea
			pair := source[color*temporalPatchFrames*patchArea:]
			copy(first[destination:destination+patchArea], pair[:patchArea])
			copy(second[destination:destination+patchArea], pair[patchArea:temporalPatchFrames*patchArea])
		}
	}
	return first, second, temporalWidth, nil
}
