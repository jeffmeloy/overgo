package projector

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func (r *PaddleOCRRunner) encodeGraph(ctx context.Context, input PaddleOCRImage) (PaddleOCROutput, error) {
	rows := input.GridH * input.GridW
	patchWidth := 3 * r.spec.PatchSize * r.spec.PatchSize
	if rows <= 0 || len(input.PixelValues) != rows*patchWidth {
		return PaddleOCROutput{}, errors.New("projector: PaddleOCR input shape is inconsistent")
	}
	if input.GridH%r.spec.MergeSize != 0 || input.GridW%r.spec.MergeSize != 0 {
		return PaddleOCROutput{}, errors.New("projector: PaddleOCR input is not merge aligned")
	}
	mergePlan, err := newPixelMergePlan(input.GridH, input.GridW, r.spec.MergeSize)
	if err != nil {
		return PaddleOCROutput{}, err
	}
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[pixels] = reference.Value{Shape: pixels.Shape, Data: input.PixelValues}
	weight := graph.weight
	patch := builder.Reshape(weight("v.patch_embd.weight"), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := graph.addOptionalBias(builder.MulMat(patch, pixels), "v.patch_embd.bias")
	rowOrder, columnOrder := paddleOCRGrid(input.GridH, input.GridW)
	hidden = r.paddleOCRPositionGraph(builder, hidden, weight("v.position_embd.weight"), input, rowOrder, columnOrder, hostFeeds)
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.pre_ln.weight"), weight("v.pre_ln.bias"), r.spec.LayerNormEpsilon)
	}
	positionsY := make([]uint32, rows)
	positionsX := make([]uint32, rows)
	for index := range positionsY {
		positionsY[index] = uint32(rowOrder[index])
		positionsX[index] = uint32(columnOrder[index])
	}
	for layer := 0; layer < r.spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.AffineLayerNorm(hidden, weight(prefix+"ln1.weight"), weight(prefix+"ln1.bias"), r.spec.LayerNormEpsilon)
		var qkv *tensor.Tensor
		if r.spec.FusedQKV[layer] {
			qkv = graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), prefix+"attn_qkv.bias")
		} else {
			parts := make([]*tensor.Tensor, 3)
			for index, part := range []string{"q", "k", "v"} {
				parts[index] = graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_"+part+".weight"), norm), prefix+"attn_"+part+".bias")
			}
			qkv = builder.Concat(builder.Concat(parts[0], parts[1], 0), parts[2], 0)
		}
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, 0, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(2*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		q = qwen3VLVisionRoPE(builder, q, positionsY, positionsX)
		k = qwen3VLVisionRoPE(builder, k, positionsY, positionsX)
		attention := builder.Attention(q, k, v, float32(1/math.Sqrt(float64(headWidth))), false)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_out.weight"), attention), prefix+"attn_out.bias")
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), prefix+"ffn_up.bias")
		up = r.paddleOCRActivationGraph(builder, up, hostFeeds)
		down := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.post_ln.weight"), weight("v.post_ln.bias"), r.spec.LayerNormEpsilon)
	}
	hidden = builder.AffineLayerNorm(hidden, weight("mm.input_norm.weight"), weight("mm.input_norm.bias"), 1e-5)
	merged := mergePlan.graph(builder, hidden)
	fc1 := builder.Add(builder.MulMat(weight("mm.1.weight"), merged), weight("mm.1.bias"))
	fc1 = r.paddleOCRActivationGraph(builder, fc1, hostFeeds)
	output := builder.Add(builder.MulMat(weight("mm.2.weight"), fc1), weight("mm.2.bias"))
	results, err := graph.execute(output)
	if err != nil {
		return PaddleOCROutput{}, fmt.Errorf("projector: execute PaddleOCR graph: %w", err)
	}
	return PaddleOCROutput{
		Embeddings: results[output], GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize,
	}, nil
}

func (r *PaddleOCRRunner) paddleOCRPositionGraph(
	builder *tensor.Builder,
	hidden, table *tensor.Tensor,
	input PaddleOCRImage,
	rowOrder, columnOrder []int,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	rows := input.GridH * input.GridW
	side := r.spec.ImageSize / r.spec.PatchSize
	indexes := [4][]uint32{}
	weights := [4][]float32{}
	for corner := range indexes {
		indexes[corner] = make([]uint32, rows)
		weights[corner] = make([]float32, rows)
	}
	coordinate := func(index, extent int) float64 {
		if extent == 1 {
			return 0
		}
		return float64(side-1) * float64(index) / float64(extent-1)
	}
	for token := 0; token < rows; token++ {
		y := coordinate(rowOrder[token], input.GridH)
		x := coordinate(columnOrder[token], input.GridW)
		y0, x0 := int(math.Floor(y)), int(math.Floor(x))
		y1, x1 := min(y0+1, side-1), min(x0+1, side-1)
		wy, wx := y-float64(y0), x-float64(x0)
		cornerIndexes := [4]int{y0*side + x0, y0*side + x1, y1*side + x0, y1*side + x1}
		cornerWeights := [4]float64{(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx}
		for corner := range indexes {
			indexes[corner][token] = uint32(cornerIndexes[corner])
			weights[corner][token] = float32(cornerWeights[corner])
		}
	}
	var position *tensor.Tensor
	for corner := range indexes {
		factor := builder.Input(fmt.Sprintf("paddle_position_weight.%d", corner), dtype.F32, tensor.MustShape(1, uint64(rows)))
		hostFeeds[factor] = reference.Value{Shape: factor.Shape, Data: weights[corner]}
		part := builder.Multiply(builder.GetRows(table, indexes[corner]), factor)
		if position == nil {
			position = part
		} else {
			position = builder.Add(position, part)
		}
	}
	return builder.Add(hidden, position)
}

func (r *PaddleOCRRunner) paddleOCRActivationGraph(
	builder *tensor.Builder,
	input *tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	switch r.spec.Activation {
	case paddleOCRGELU:
		return qwen3VLGELUTanh(builder, input, hostFeeds)
	case paddleOCRSiLU:
		return builder.SiLU(input)
	default:
		return builder.Multiply(input, builder.Sigmoid(builder.Scale(input, 1.702)))
	}
}
