package projector

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/tensor"
)

type rasterInterpolation uint8

const (
	rasterBicubic rasterInterpolation = iota
	rasterBilinear
)

type rowStorage struct {
	elements int
	width    int
}

func validateGridStorage(height, width int, storage ...rowStorage) (int, error) {
	rows, ok := checked.MulInt(height, width)
	if !ok || !checked.PositiveInts(rows) {
		return tensor.FirstOffset, errors.New("projector: invalid grid geometry")
	}
	if err := validateRowStorage(rows, storage...); err != nil {
		return tensor.FirstOffset, err
	}
	return rows, nil
}

func validateRowStorage(rows int, storage ...rowStorage) error {
	if !checked.PositiveInts(rows) {
		return errors.New("projector: invalid row count")
	}
	for _, binding := range storage {
		expected, ok := checked.MulInt(rows, binding.width)
		if !ok || !checked.PositiveInts(binding.width) || binding.elements != expected {
			return errors.New("projector: row storage is inconsistent")
		}
	}
	return nil
}

func validateSpatialPatchStorage(elements, gridH, gridW, patchSize, channels int) (int, int, error) {
	rows, rowsOK := checked.MulInt(gridH, gridW)
	patchArea, areaOK := checked.MulInt(patchSize, patchSize)
	width, widthOK := checked.MulInt(patchArea, channels)
	if !rowsOK || !areaOK || !widthOK {
		return tensor.FirstOffset, tensor.FirstOffset, errors.New("projector: spatial patch extents overflow")
	}
	if err := validateRowStorage(rows, rowStorage{elements: elements, width: width}); err != nil {
		return tensor.FirstOffset, tensor.FirstOffset, err
	}
	return rows, width, nil
}

type rasterPatchPlan struct {
	patchSize     int
	mergeSize     int
	defaultBudget pixelBudget
	mean          [media.RGBChannels]float32
	std           [media.RGBChannels]float32
	interpolation rasterInterpolation
}

func preprocessRasterPatches(source image.Image, plan rasterPatchPlan, options RasterPatchOptions) (RasterPatchImage, error) {
	if source == nil {
		return RasterPatchImage{}, errors.New("projector: image is nil")
	}
	if !checked.PositiveInts(plan.patchSize, plan.mergeSize) {
		return RasterPatchImage{}, errors.New("projector: invalid raster patch geometry")
	}
	for channel := range plan.std {
		if !checked.PositiveFinite32(plan.std[channel]) {
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
		resized = media.ResizeBicubic(source, resizedW, resizedH)
	case rasterBilinear:
		resized = resizeImageBilinear(source, resizedW, resizedH)
	default:
		return RasterPatchImage{}, errors.New("projector: invalid raster interpolation")
	}
	return patchRasterImage(resized, plan.patchSize, plan.mean, plan.std)
}

func patchRasterImage(source image.Image, patchSize int, mean, std [media.RGBChannels]float32) (RasterPatchImage, error) {
	bounds := source.Bounds()
	gridH, heightOK := checked.DivExactInt(bounds.Dy(), patchSize)
	gridW, widthOK := checked.DivExactInt(bounds.Dx(), patchSize)
	if !heightOK || !widthOK || !checked.PositiveInts(gridH, gridW) {
		return RasterPatchImage{}, errors.New("projector: raster is not patch aligned")
	}
	patchArea := patchSize * patchSize
	patchWidth := media.RGBChannels * patchArea
	pixels := make([]float32, gridH*gridW*patchWidth)
	for patchY := range gridH {
		for patchX := range gridW {
			row := (patchY*gridW + patchX) * patchWidth
			for y := range patchSize {
				for x := range patchSize {
					red, green, blue, _ := source.At(bounds.Min.X+patchX*patchSize+x, bounds.Min.Y+patchY*patchSize+y).RGBA()
					values := [media.RGBChannels]uint32{red, green, blue}
					pixel := y*patchSize + x
					for channel, value := range values {
						pixels[row+channel*patchArea+pixel] = (media.NormalizedRGBAChannel(value) - mean[channel]) / std[channel]
					}
				}
			}
		}
	}
	return RasterPatchImage{PixelValues: pixels, GridH: gridH, GridW: gridW}, nil
}

func resizeFitBicubic(source image.Image, width, height int, fill color.Color) image.Image {
	bounds := source.Bounds()
	scale := math.Min(float64(width)/float64(bounds.Dx()), float64(height)/float64(bounds.Dy()))
	resizedW := max(tensor.SingletonExtent, min(width, int(math.Ceil(float64(bounds.Dx())*scale))))
	resizedH := max(tensor.SingletonExtent, min(height, int(math.Ceil(float64(bounds.Dy())*scale))))
	resized := media.ResizeBicubic(source, resizedW, resizedH)
	output := image.NewRGBA(image.Rect(tensor.FirstOffset, tensor.FirstOffset, width, height))
	if fill != nil {
		for y := range height {
			for x := range width {
				output.Set(x, y, fill)
			}
		}
	}
	offsetX, offsetY := (width-resizedW)/tensor.PairedExtent, (height-resizedH)/tensor.PairedExtent
	for y := range resizedH {
		for x := range resizedW {
			output.Set(offsetX+x, offsetY+y, resized.At(x, y))
		}
	}
	return output
}

func resizeImageBilinear(source image.Image, width, height int) *image.RGBA {
	bounds := source.Bounds()
	inputW, inputH := bounds.Dx(), bounds.Dy()
	output := image.NewRGBA(image.Rect(tensor.FirstOffset, tensor.FirstOffset, width, height))
	for y := range height {
		sourceY := (float64(y)+media.RasterSampleCenter)*float64(inputH)/float64(height) - media.RasterSampleCenter
		y0 := max(tensor.FirstOffset, min(inputH-tensor.SingletonExtent, int(math.Floor(sourceY))))
		y1 := min(y0+tensor.SingletonExtent, inputH-tensor.SingletonExtent)
		wy := sourceY - math.Floor(sourceY)
		if sourceY < tensor.FirstOffset {
			wy = tensor.FirstOffset
		}
		for x := range width {
			sourceX := (float64(x)+media.RasterSampleCenter)*float64(inputW)/float64(width) - media.RasterSampleCenter
			x0 := max(tensor.FirstOffset, min(inputW-tensor.SingletonExtent, int(math.Floor(sourceX))))
			x1 := min(x0+tensor.SingletonExtent, inputW-tensor.SingletonExtent)
			wx := sourceX - math.Floor(sourceX)
			if sourceX < tensor.FirstOffset {
				wx = tensor.FirstOffset
			}
			corners := [tensor.MaxDimensions][tensor.MaxDimensions]uint32{}
			corners[tensor.FirstOffset][tensor.FirstOffset], corners[tensor.FirstOffset][tensor.SingletonExtent], corners[tensor.FirstOffset][tensor.PairedExtent], corners[tensor.FirstOffset][tensor.TripleExtent] = source.At(bounds.Min.X+x0, bounds.Min.Y+y0).RGBA()
			corners[tensor.SingletonExtent][tensor.FirstOffset], corners[tensor.SingletonExtent][tensor.SingletonExtent], corners[tensor.SingletonExtent][tensor.PairedExtent], corners[tensor.SingletonExtent][tensor.TripleExtent] = source.At(bounds.Min.X+x1, bounds.Min.Y+y0).RGBA()
			corners[tensor.PairedExtent][tensor.FirstOffset], corners[tensor.PairedExtent][tensor.SingletonExtent], corners[tensor.PairedExtent][tensor.PairedExtent], corners[tensor.PairedExtent][tensor.TripleExtent] = source.At(bounds.Min.X+x0, bounds.Min.Y+y1).RGBA()
			corners[tensor.TripleExtent][tensor.FirstOffset], corners[tensor.TripleExtent][tensor.SingletonExtent], corners[tensor.TripleExtent][tensor.PairedExtent], corners[tensor.TripleExtent][tensor.TripleExtent] = source.At(bounds.Min.X+x1, bounds.Min.Y+y1).RGBA()
			unit := float64(tensor.SingletonExtent)
			weights := [tensor.MaxDimensions]float64{(unit - wx) * (unit - wy), wx * (unit - wy), (unit - wx) * wy, wx * wy}
			index := output.PixOffset(x, y)
			for channel := range media.RGBChannels {
				var value float64
				for corner := range corners {
					value += weights[corner] * float64(media.RGBAChannel8(corners[corner][channel]))
				}
				output.Pix[index+channel] = media.RoundedUint8(value)
			}
			output.Pix[index+media.RGBChannels] = ^uint8(tensor.FirstOffset)
		}
	}
	return output
}

type pixelMergePlan struct {
	inputRows                 int
	outputRows                int
	outputHeight, outputWidth int
	indexSets                 [][]uint32
}

func newPixelMergePlan(height, width, merge int) (pixelMergePlan, error) {
	blocksY, heightOK := checked.DivExactInt(height, merge)
	blocksX, widthOK := checked.DivExactInt(width, merge)
	if !heightOK || !widthOK || !checked.PositiveInts(blocksY, blocksX) {
		return pixelMergePlan{}, errors.New("projector: invalid pixel merge geometry")
	}
	inputRows, ok := checked.MulInt(height, width)
	if !ok || uint64(inputRows) > uint64(math.MaxUint32)+tensor.SingletonExtent {
		return pixelMergePlan{}, errors.New("projector: pixel merge input size overflow")
	}
	indexCount, ok := checked.MulInt(merge, merge)
	if !ok {
		return pixelMergePlan{}, errors.New("projector: pixel merge size overflow")
	}
	outputRows, ok := checked.MulInt(blocksY, blocksX)
	if !ok {
		return pixelMergePlan{}, errors.New("projector: pixel merge output size overflow")
	}
	indexSets := make([][]uint32, indexCount)
	for index := range indexSets {
		indexSets[index] = make([]uint32, tensor.FirstOffset, outputRows)
	}
	for blockY := range blocksY {
		for blockX := range blocksX {
			for y := range merge {
				for x := range merge {
					offset := y*merge + x
					indexSets[offset] = append(indexSets[offset], uint32((blockY*merge+y)*width+blockX*merge+x))
				}
			}
		}
	}
	return pixelMergePlan{
		inputRows: inputRows, outputRows: outputRows,
		outputHeight: blocksY, outputWidth: blocksX, indexSets: indexSets,
	}, nil
}

func (plan pixelMergePlan) graph(builder *tensor.Builder, input *tensor.Tensor) *tensor.Tensor {
	output := builder.GetRows(input, plan.indexSets[tensor.FirstOffset])
	for offset := tensor.SingletonExtent; offset < len(plan.indexSets); offset++ {
		output = builder.Concat(output, builder.GetRows(input, plan.indexSets[offset]), tensor.FirstOffset)
	}
	return output
}

func (plan pixelMergePlan) shuffle(values []float32, width int) ([]float32, error) {
	if !checked.PositiveInts(width) {
		return nil, fmt.Errorf("projector: pixel merge input has %d values for %d rows at width %d", len(values), plan.inputRows, width)
	}
	expected, ok := checked.MulInt(plan.inputRows, width)
	if !ok || len(values) != expected {
		return nil, fmt.Errorf("projector: pixel merge input has %d values for %d rows at width %d", len(values), plan.inputRows, width)
	}
	outputSize, ok := checked.MulInt(plan.outputRows, width)
	if ok {
		outputSize, ok = checked.MulInt(outputSize, len(plan.indexSets))
	}
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
	if !checked.PositiveInts(rows, patchArea) {
		return nil, nil, tensor.FirstOffset, errors.New("projector: invalid temporal patch geometry")
	}
	temporalElements, ok := checked.Mul64(uint64(rows), uint64(patchArea))
	if !ok {
		return nil, nil, tensor.FirstOffset, errors.New("projector: temporal patch size overflow")
	}
	temporalElements, ok = checked.Mul64(temporalElements, media.RGBChannels*tensor.PairedExtent)
	if !ok {
		return nil, nil, tensor.FirstOffset, errors.New("projector: temporal patch size overflow")
	}
	expected, ok := checked.Int(temporalElements)
	if !ok {
		return nil, nil, tensor.FirstOffset, errors.New("projector: temporal patch size overflow")
	}
	if len(values) != expected {
		return nil, nil, tensor.FirstOffset, fmt.Errorf("projector: temporal patch tensor has %d values, want %d", len(values), expected)
	}
	temporalWidth := media.RGBChannels * patchArea
	first := make([]float32, rows*temporalWidth)
	second := make([]float32, rows*temporalWidth)
	for row := range rows {
		source := values[row*temporalWidth*tensor.PairedExtent:]
		for color := range media.RGBChannels {
			destination := row*temporalWidth + color*patchArea
			pair := source[color*tensor.PairedExtent*patchArea:]
			copy(first[destination:destination+patchArea], pair[:patchArea])
			copy(second[destination:destination+patchArea], pair[patchArea:tensor.PairedExtent*patchArea])
		}
	}
	return first, second, temporalWidth, nil
}
