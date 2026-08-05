package projector

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
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

func pointwiseConvolution() convolutionPlan {
	return convolutionPlan{strideX: 1, strideY: 1}
}

func sameConvolution(input, weight *tensor.Tensor, stride uint32, depthwise bool) convolutionPlan {
	kernelW, kernelH := uint32(weight.Shape.Dims[0]), uint32(weight.Shape.Dims[1])
	width, height := uint32(input.Shape.Dims[1]), uint32(input.Shape.Dims[2])
	outputW, outputH := (width+stride-1)/stride, (height+stride-1)/stride
	paddingW := uint32(max(int((outputW-1)*stride+kernelW-width), 0))
	paddingH := uint32(max(int((outputH-1)*stride+kernelH-height), 0))
	return convolutionPlan{
		strideX: stride, strideY: stride,
		padLeft: paddingW / 2, padRight: paddingW - paddingW/2,
		padTop: paddingH / 2, padBottom: paddingH - paddingH/2,
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

type visionAttentionPlan struct {
	hidden int
	heads  int
	causal bool
}

func newVisionAttentionPlan(hidden, heads int, causal bool) (visionAttentionPlan, error) {
	if hidden <= 0 || heads <= 0 || hidden%heads != 0 {
		return visionAttentionPlan{}, errors.New("projector: invalid attention dimensions")
	}
	return visionAttentionPlan{hidden: hidden, heads: heads, causal: causal}, nil
}

func mustVisionAttentionPlan(hidden, heads int, causal bool) visionAttentionPlan {
	plan, err := newVisionAttentionPlan(hidden, heads, causal)
	if err != nil {
		panic(err)
	}
	return plan
}

func (p visionAttentionPlan) headWidth() int {
	return p.hidden / p.heads
}

func (p visionAttentionPlan) scale() float32 {
	return float32(1 / math.Sqrt(float64(p.headWidth())))
}

func (p visionAttentionPlan) graph(
	builder *tensor.Builder,
	query, key, value *tensor.Tensor,
) *tensor.Tensor {
	return builder.Attention(query, key, value, p.scale(), p.causal)
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
	if p.MinPixels <= 0 || p.MaxPixels < p.MinPixels || p.MaxAspectRatio <= 0 {
		return errors.New("projector: invalid pixel budget")
	}
	return nil
}

func (p pixelBudget) resize(height, width, factor int) (int, int, error) {
	if err := p.validate(); err != nil {
		return 0, 0, err
	}
	return smartResizeAligned(
		height, width, factor, p.MinPixels, p.MaxPixels, p.MaxAspectRatio,
	)
}
