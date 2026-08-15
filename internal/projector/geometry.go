package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

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
	rotaryPairWidth                    = 2
	visionRoPEAxisCount                = 2
	visionRoPEComponentCount           = rotaryPairWidth * visionRoPEAxisCount
	rgba16To8Shift                     = 8
	maxUint8Channel                    = 255
	opaqueAlpha                        = 255
)

func normalizedImageChannel(value uint32) float32 {
	return float32(value>>rgba16To8Shift) / maxUint8Channel
}

type rasterInterpolation uint8

const (
	rasterBicubic rasterInterpolation = iota
	rasterBilinear
)

type rasterPatchPlan struct {
	patchSize     int
	mergeSize     int
	defaultBudget pixelBudget
	mean          [rgbChannelCount]float32
	std           [rgbChannelCount]float32
	interpolation rasterInterpolation
}

func encodeRasterPatches[Output any](ctx context.Context, source image.Image, options RasterPatchOptions, plan rasterPatchPlan, validate func() error, encode func(context.Context, RasterPatchImage) (Output, error)) (Output, error) {
	var zero Output
	if err := validate(); err != nil {
		return zero, err
	}
	input, err := preprocessRasterPatches(source, plan, options)
	if err != nil {
		return zero, err
	}
	return encode(ctx, input)
}

func preprocessRasterPatches(source image.Image, plan rasterPatchPlan, options RasterPatchOptions) (RasterPatchImage, error) {
	if source == nil {
		return RasterPatchImage{}, errors.New("projector: image is nil")
	}
	if plan.patchSize <= 0 || plan.mergeSize <= 0 {
		return RasterPatchImage{}, errors.New("projector: invalid raster patch geometry")
	}
	for channel := range plan.std {
		if plan.std[channel] <= 0 {
			return RasterPatchImage{}, fmt.Errorf("projector: invalid raster normalization channel %d", channel)
		}
	}
	budget := pixelBudget(options)
	if budget == (pixelBudget{}) {
		budget = plan.defaultBudget
	}
	bounds := source.Bounds()
	resizedH, resizedW, err := budget.resize(bounds.Dy(), bounds.Dx(), plan.patchSize*plan.mergeSize)
	if err != nil {
		return RasterPatchImage{}, err
	}
	var resized *image.RGBA
	switch plan.interpolation {
	case rasterBicubic:
		resized = resizeImageBicubic(source, resizedW, resizedH)
	case rasterBilinear:
		resized = resizeImageBilinear(source, resizedW, resizedH)
	default:
		return RasterPatchImage{}, errors.New("projector: invalid raster interpolation")
	}
	gridH, gridW := resizedH/plan.patchSize, resizedW/plan.patchSize
	patchArea := plan.patchSize * plan.patchSize
	patchWidth := rgbChannelCount * patchArea
	pixels := make([]float32, gridH*gridW*patchWidth)
	for patchY := 0; patchY < gridH; patchY++ {
		for patchX := 0; patchX < gridW; patchX++ {
			row := (patchY*gridW + patchX) * patchWidth
			for y := 0; y < plan.patchSize; y++ {
				for x := 0; x < plan.patchSize; x++ {
					red, green, blue, _ := resized.At(patchX*plan.patchSize+x, patchY*plan.patchSize+y).RGBA()
					values := [rgbChannelCount]uint32{red, green, blue}
					pixel := y*plan.patchSize + x
					for channel, value := range values {
						pixels[row+channel*patchArea+pixel] = (normalizedImageChannel(value) - plan.mean[channel]) / plan.std[channel]
					}
				}
			}
		}
	}
	return RasterPatchImage{PixelValues: pixels, GridH: gridH, GridW: gridW}, nil
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
