package projector

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type convolutionPlan struct {
	strideX   uint32
	strideY   uint32
	padLeft   uint32
	padRight  uint32
	padTop    uint32
	padBottom uint32
	depthwise bool
}

func sameConvolution(input, weight *tensor.Tensor, stride uint32, depthwise bool) convolutionPlan {
	kernelW, kernelH := uint32(weight.Shape.Dims[tensor.FirstOffset]), uint32(weight.Shape.Dims[tensor.SingletonExtent])
	width, height := uint32(input.Shape.Dims[tensor.SingletonExtent]), uint32(input.Shape.Dims[tensor.PairedExtent])
	outputW := (width + stride - tensor.SingletonExtent) / stride
	outputH := (height + stride - tensor.SingletonExtent) / stride
	paddingW := uint32(max(int((outputW-tensor.SingletonExtent)*stride+kernelW-width), tensor.FirstOffset))
	paddingH := uint32(max(int((outputH-tensor.SingletonExtent)*stride+kernelH-height), tensor.FirstOffset))
	return convolutionPlan{
		strideX: stride, strideY: stride,
		padLeft: paddingW / tensor.PairedExtent, padRight: paddingW - paddingW/tensor.PairedExtent,
		padTop: paddingH / tensor.PairedExtent, padBottom: paddingH - paddingH/tensor.PairedExtent,
		depthwise: depthwise,
	}
}

func (p convolutionPlan) graph(
	builder *tensor.Builder,
	input, weight, bias *tensor.Tensor,
) *tensor.Tensor {
	return builder.Conv2D(
		input, weight, bias, p.strideX, p.strideY,
		p.padLeft, p.padRight, p.padTop, p.padBottom, p.depthwise,
	)
}

func convolveSame(
	builder *tensor.Builder,
	input, weight, bias *tensor.Tensor,
	stride uint32,
	depthwise bool,
) *tensor.Tensor {
	return sameConvolution(input, weight, stride, depthwise).graph(builder, input, weight, bias)
}

func convolveCentered(
	builder *tensor.Builder,
	input, weight, bias *tensor.Tensor,
	stride uint32,
	depthwise bool,
) *tensor.Tensor {
	padX := uint32(weight.Shape.Dims[tensor.FirstOffset] / tensor.PairedExtent)
	padY := uint32(weight.Shape.Dims[tensor.SingletonExtent] / tensor.PairedExtent)
	return builder.Conv2D(input, weight, bias, stride, stride, padX, padX, padY, padY, depthwise)
}

func scaleChannels(builder *tensor.Builder, input, weight *tensor.Tensor) *tensor.Tensor {
	return builder.Multiply(input, builder.Reshape(weight, input.Shape.Dims[tensor.FirstOffset], tensor.SingletonExtent, tensor.SingletonExtent))
}

func resizeSpatialNearest(builder *tensor.Builder, input *tensor.Tensor, width, height int) *tensor.Tensor {
	inputW, inputH := int(input.Shape.Dims[tensor.SingletonExtent]), int(input.Shape.Dims[tensor.PairedExtent])
	if inputW == width && inputH == height {
		return input
	}
	indices := make([]uint32, width*height)
	for y := range height {
		for x := range width {
			indices[x+width*y] = uint32(x*inputW/width + inputW*(y*inputH/height))
		}
	}
	flat := builder.Reshape(input, input.Shape.Dims[tensor.FirstOffset], uint64(inputW*inputH))
	return builder.Reshape(builder.GetRows(flat, indices), input.Shape.Dims[tensor.FirstOffset], uint64(width), uint64(height))
}

func averagePoolSpatial(builder *tensor.Builder, input *tensor.Tensor, width, height int) *tensor.Tensor {
	inputW, inputH := int(input.Shape.Dims[tensor.SingletonExtent]), int(input.Shape.Dims[tensor.PairedExtent])
	if inputW == width && inputH == height {
		return input
	}
	factorW, exactW := checked.DivExact64(uint64(inputW), uint64(width))
	factorH, exactH := checked.DivExact64(uint64(inputH), uint64(height))
	if !exactW || !exactH {
		builder.Reshape(input, tensor.FirstOffset)
		return nil
	}
	flat := builder.Reshape(input, input.Shape.Dims[tensor.FirstOffset], uint64(inputW*inputH))
	indices := make([]uint32, width*height)
	var result *tensor.Tensor
	for dy := range int(factorH) {
		for dx := range int(factorW) {
			for y := range height {
				for x := range width {
					indices[x+width*y] = uint32(x*int(factorW) + dx + inputW*(y*int(factorH)+dy))
				}
			}
			part := builder.GetRows(flat, indices)
			if result == nil {
				result = part
			} else {
				result = builder.Add(result, part)
			}
		}
	}
	result = builder.Scale(result, float32(tensor.SingletonExtent)/float32(factorW*factorH))
	return builder.Reshape(result, input.Shape.Dims[tensor.FirstOffset], uint64(width), uint64(height))
}

type visionAttentionPlan struct {
	hidden int
	heads  int
	causal bool
}

func compileVisionAttention(hidden, heads int) visionAttentionPlan {
	return visionAttentionPlan{hidden: hidden, heads: heads}
}

func (p visionAttentionPlan) headWidth() int {
	return p.hidden / p.heads
}

func (p visionAttentionPlan) scale() float32 {
	return hostmath.InvSqrt32(uint64(p.headWidth()))
}

func (p visionAttentionPlan) graph(
	builder *tensor.Builder,
	query, key, value *tensor.Tensor,
) *tensor.Tensor {
	return builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{Scale: p.scale(), Causal: p.causal})
}

type normalizationKind uint8

const (
	normalizationAffineLayer normalizationKind = iota
	normalizationWeightedRMS
	normalizationRMS
)

type normalizationPlan struct {
	kind    normalizationKind
	epsilon float32
	bf16    bool
}

func (p normalizationPlan) graph(
	builder *tensor.Builder,
	input, weight, bias *tensor.Tensor,
) *tensor.Tensor {
	var output *tensor.Tensor
	switch p.kind {
	case normalizationAffineLayer:
		output = builder.AffineLayerNorm(input, weight, bias, p.epsilon)
	case normalizationWeightedRMS:
		output = builder.WeightedRMSNorm(input, weight, p.epsilon)
	default:
		output = builder.RMSNorm(input, p.epsilon)
	}
	if p.bf16 {
		output = builder.BF16Round(output)
	}
	return output
}

func (p pixelBudget) validate() error {
	if !checked.PositiveInts(p.MinPixels) || p.MaxPixels < p.MinPixels {
		return errors.New("projector: invalid pixel budget")
	}
	return nil
}

func (p pixelBudget) resize(height, width, factor int) (int, int, error) {
	if err := p.validate(); err != nil {
		return tensor.FirstOffset, tensor.FirstOffset, err
	}
	return media.ResizeAligned(height, width, factor, p.MinPixels, p.MaxPixels)
}

func spatialWeightedRMSNorm(builder *tensor.Builder, input, weight *tensor.Tensor, epsilon float32) *tensor.Tensor {
	channels := input.Shape.Dims[tensor.FirstOffset]
	width := input.Shape.Dims[tensor.SingletonExtent]
	height := input.Shape.Dims[tensor.PairedExtent]
	flat := builder.Reshape(input, channels, width*height)
	normalized := builder.WeightedRMSNorm(flat, builder.Reshape(weight, channels), epsilon)
	return builder.Reshape(normalized, channels, width, height)
}

func gridCoordinates(height, width int) ([]int, []int) {
	rows, columns := make([]int, height*width), make([]int, height*width)
	for index := range rows {
		rows[index], columns[index] = index/width, index%width
	}
	return rows, columns
}

func repeatCoordinates(rows, columns []int, count int) ([]uint32, []uint32, int) {
	storage := make([]uint32, tensor.PairedExtent*count)
	repeatedRows, repeatedColumns := storage[:count], storage[count:]
	for index := range count {
		repeatedRows[index] = uint32(rows[index%len(rows)])
		repeatedColumns[index] = uint32(columns[index%len(columns)])
	}
	return repeatedRows, repeatedColumns, len(rows)
}

func spatialPositionGraph(
	builder *tensor.Builder,
	hidden, table *tensor.Tensor,
	rows, spatial, gridH, gridW, tableSide int,
	rowOrder, columnOrder []int,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	indexes := make([]uint32, tensor.MaxDimensions*rows)
	weights := make([]float32, tensor.MaxDimensions*rows)
	coordinate := func(index, extent int) float64 {
		if extent == tensor.SingletonExtent {
			return tensor.FirstOffset
		}
		return float64(tableSide-tensor.SingletonExtent) * float64(index) / float64(extent-tensor.SingletonExtent)
	}
	for row := range rows {
		token := row % spatial
		y, x := coordinate(rowOrder[token], gridH), coordinate(columnOrder[token], gridW)
		y0, x0 := int(math.Floor(y)), int(math.Floor(x))
		y1 := min(y0+tensor.SingletonExtent, tableSide-tensor.SingletonExtent)
		x1 := min(x0+tensor.SingletonExtent, tableSide-tensor.SingletonExtent)
		wy, wx := y-float64(y0), x-float64(x0)
		cornerIndexes := [tensor.MaxDimensions]int{y0*tableSide + x0, y0*tableSide + x1, y1*tableSide + x0, y1*tableSide + x1}
		unit := float64(tensor.SingletonExtent)
		cornerWeights := [tensor.MaxDimensions]float64{(unit - wy) * (unit - wx), (unit - wy) * wx, wy * (unit - wx), wy * wx}
		for corner := range cornerIndexes {
			indexes[corner*rows+row] = uint32(cornerIndexes[corner])
			weights[corner*rows+row] = float32(cornerWeights[corner])
		}
	}
	var position *tensor.Tensor
	for corner := range tensor.MaxDimensions {
		start := corner * rows
		factor := builder.Input(fmt.Sprintf("position_weight.%d", corner), dtype.F32, tensor.MustShape(tensor.SingletonExtent, uint64(rows)))
		hostFeeds[factor] = reference.Value{Shape: factor.Shape, Data: weights[start : start+rows]}
		part := builder.Multiply(builder.GetRows(table, indexes[start:start+rows]), factor)
		if position == nil {
			position = part
		} else {
			position = builder.Add(position, part)
		}
	}
	return builder.Add(hidden, position)
}
