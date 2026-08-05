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

func (r *HunyuanVLRunner) encodeGraph(ctx context.Context, input HunyuanVLImage) (HunyuanVLOutput, error) {
	rows := input.GridH * input.GridW
	patchWidth := 3 * r.spec.PatchSize * r.spec.PatchSize
	if rows <= 0 || input.GridH%r.spec.MergeSize != 0 || input.GridW%r.spec.MergeSize != 0 || len(input.PixelValues) != rows*patchWidth {
		return HunyuanVLOutput{}, errors.New("projector: Hunyuan-VL input shape is inconsistent")
	}
	mergePlan, err := newPixelMergePlan(input.GridH, input.GridW, r.spec.MergeSize)
	if err != nil {
		return HunyuanVLOutput{}, err
	}
	conv0, err := loadProjectorHostTensor(ctx, r.file, "mm.0.weight")
	if err != nil {
		return HunyuanVLOutput{}, err
	}
	mergeFactor := r.spec.MergeSize * r.spec.MergeSize
	convWidth := r.spec.Hidden * mergeFactor
	reordered := make([]float32, len(conv0.Data))
	for output := 0; output < r.spec.ConvIntermediate; output++ {
		for offset := 0; offset < mergeFactor; offset++ {
			for channel := 0; channel < r.spec.Hidden; channel++ {
				source := output*convWidth + channel*mergeFactor + offset
				destination := output*convWidth + offset*r.spec.Hidden + channel
				reordered[destination] = conv0.Data[source]
			}
		}
	}
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	conv0Input := builder.Input("mm.0.weight.reordered", dtype.F32, tensor.MustShape(uint64(convWidth), uint64(r.spec.ConvIntermediate)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[pixels] = pixelsValue(pixels, input.PixelValues)
	hostFeeds[conv0Input] = pixelsValue(conv0Input, reordered)
	weight := graph.weight
	patch := builder.Reshape(weight("v.patch_embd.weight"), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := graph.addOptionalBias(builder.MulMat(patch, pixels), "v.patch_embd.bias")
	hidden = r.hunyuanVLPositionGraph(builder, hidden, weight("v.position_embd.weight"), input.GridH, input.GridW, hostFeeds)
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.pre_ln.weight"), weight("v.pre_ln.bias"), r.spec.LayerNormEpsilon)
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
		attention := mustVisionAttentionPlan(r.spec.Hidden, r.spec.Heads, false).graph(builder, q, k, v)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_out.weight"), attention), prefix+"attn_out.bias")
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), prefix+"ffn_up.bias")
		up = qwen3VLGELUTanh(builder, up, hostFeeds)
		down := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.post_ln.weight"), weight("v.post_ln.bias"), r.spec.LayerNormEpsilon)
	}
	hidden = builder.WeightedRMSNorm(hidden, weight("mm.pre_norm.weight"), r.spec.LayerNormEpsilon)
	mergedH, mergedW := input.GridH/r.spec.MergeSize, input.GridW/r.spec.MergeSize
	merged := mergePlan.graph(builder, hidden)
	projected := builder.Add(builder.MulMat(conv0Input, merged), weight("mm.0.bias"))
	projected = qwen3VLGELUTanh(builder, projected, hostFeeds)
	conv2 := builder.Reshape(weight("mm.2.weight"), uint64(r.spec.ConvIntermediate), uint64(r.spec.ProjectorInput))
	projected = builder.Add(builder.MulMat(conv2, projected), weight("mm.2.bias"))
	newline := builder.Reshape(weight("v.image_newline"), uint64(r.spec.ProjectorInput), 1)
	var content *tensor.Tensor
	for y := 0; y < mergedH; y++ {
		row := builder.FlatSlice(projected, uint64(y*mergedW*r.spec.ProjectorInput), uint64(r.spec.ProjectorInput), uint64(mergedW))
		row = builder.Concat(row, newline, 1)
		if content == nil {
			content = row
		} else {
			content = builder.Concat(content, row, 1)
		}
	}
	content = builder.Add(builder.MulMat(weight("mm.model.fc.weight"), content), weight("mm.model.fc.bias"))
	begin := builder.Reshape(weight("mm.image_begin"), uint64(r.spec.OutputHidden), 1)
	end := builder.Reshape(weight("mm.image_end"), uint64(r.spec.OutputHidden), 1)
	output := builder.Concat(builder.Concat(begin, content, 1), end, 1)
	output = builder.WeightedRMSNorm(output, weight("mm.post_norm.weight"), r.spec.LayerNormEpsilon)
	results, err := graph.execute(output)
	if err != nil {
		return HunyuanVLOutput{}, fmt.Errorf("projector: execute Hunyuan-VL graph: %w", err)
	}
	return HunyuanVLOutput{Embeddings: results[output], GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}

func pixelsValue(node *tensor.Tensor, data []float32) reference.Value {
	return reference.Value{Shape: node.Shape, Data: data}
}

func (r *HunyuanVLRunner) hunyuanVLPositionGraph(
	builder *tensor.Builder,
	hidden, table *tensor.Tensor,
	gridH, gridW int,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	rows := gridH * gridW
	side := r.spec.ImageSize / r.spec.PatchSize
	sx := (float64(gridW) + 0.1) / float64(side)
	sy := (float64(gridH) + 0.1) / float64(side)
	indexes := [4][]uint32{}
	weights := [4][]float32{}
	for corner := range indexes {
		indexes[corner] = make([]uint32, rows)
		weights[corner] = make([]float32, rows)
	}
	for y := 0; y < gridH; y++ {
		fy := (float64(y)+0.5)/sy - 0.5
		y0 := max(0, min(side-1, int(math.Floor(fy))))
		y1 := max(0, min(side-1, int(math.Floor(fy))+1))
		wy := min(1.0, max(0.0, fy-float64(y0)))
		for x := 0; x < gridW; x++ {
			fx := (float64(x)+0.5)/sx - 0.5
			x0 := max(0, min(side-1, int(math.Floor(fx))))
			x1 := max(0, min(side-1, int(math.Floor(fx))+1))
			wx := min(1.0, max(0.0, fx-float64(x0)))
			row := y*gridW + x
			cornerIndexes := [4]int{y0*side + x0, y0*side + x1, y1*side + x0, y1*side + x1}
			cornerWeights := [4]float64{(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx}
			for corner := range indexes {
				indexes[corner][row] = uint32(cornerIndexes[corner])
				weights[corner][row] = float32(cornerWeights[corner])
			}
		}
	}
	var position *tensor.Tensor
	for corner := range indexes {
		factor := builder.Input(fmt.Sprintf("hunyuan_position_weight.%d", corner), dtype.F32, tensor.MustShape(1, uint64(rows)))
		hostFeeds[factor] = pixelsValue(factor, weights[corner])
		part := builder.Multiply(builder.GetRows(table, indexes[corner]), factor)
		if position == nil {
			position = part
		} else {
			position = builder.Add(position, part)
		}
	}
	return builder.Add(hidden, position)
}
