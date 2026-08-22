package projector

import (
	"context"
	"fmt"
	"math"

	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

const interpolationExtentBias = 0.1

func (r *HunyuanVLRunner) encodeGraph(ctx context.Context, input RasterPatchImage) (gridOutput, error) {
	rows, patchWidth, err := validateSpatialPatchStorage(
		len(input.PixelValues), input.GridH, input.GridW, r.spec.PatchSize, media.RGBChannels,
	)
	if err != nil {
		return gridOutput{}, fmt.Errorf("projector: Hunyuan-VL input: %w", err)
	}
	mergePlan, err := newPixelMergePlan(input.GridH, input.GridW, r.spec.MergeSize)
	if err != nil {
		return gridOutput{}, err
	}
	conv0, err := loadProjectorHostTensor(ctx, r.file, "mm.0.weight")
	if err != nil {
		return gridOutput{}, err
	}
	mergeFactor := r.spec.MergeSize * r.spec.MergeSize
	convWidth := r.spec.Hidden * mergeFactor
	reordered := make([]float32, len(conv0.Data))
	for output := range r.spec.ConvIntermediate {
		for offset := range mergeFactor {
			for channel := range r.spec.Hidden {
				source := output*convWidth + channel*mergeFactor + offset
				destination := output*convWidth + offset*r.spec.Hidden + channel
				reordered[destination] = conv0.Data[source]
			}
		}
	}
	builder := tensor.NewBuilder()
	pixels := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	conv0Input := builder.Input("mm.0.weight.reordered", dtype.F32, tensor.MustShape(uint64(convWidth), uint64(r.spec.ConvIntermediate)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[pixels] = pixelsValue(pixels, input.PixelValues)
	hostFeeds[conv0Input] = pixelsValue(conv0Input, reordered)
	weight := graph.weight
	patch := builder.Reshape(weight(visionPatchWeightTensor), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := graph.addOptionalBias(builder.MulMat(patch, pixels), visionPatchBiasTensor)
	hidden = biasedSpatialPositionGraph(builder, hidden, weight(visionPositionWeightTensor), input.GridH, input.GridW,
		r.spec.ImageSize/r.spec.PatchSize, hostFeeds)
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight(visionPreNormWeightTensor), weight(visionPreNormBiasTensor), r.spec.LayerNormEpsilon)
	}
	for layer := range r.spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.AffineLayerNorm(hidden, weight(prefix+"ln1.weight"), weight(prefix+"ln1.bias"), r.spec.LayerNormEpsilon)
		var qkv *tensor.Tensor
		if r.spec.FusedQKV[layer] {
			qkv = graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), prefix+"attn_qkv.bias")
		} else {
			parts := make([]*tensor.Tensor, tensor.TripleExtent)
			for index, part := range [tensor.TripleExtent]string{"q", "k", "v"} {
				parts[index] = graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_"+part+".weight"), norm), prefix+"attn_"+part+".bias")
			}
			qkv = builder.Concat(builder.Concat(parts[tensor.FirstOffset], parts[tensor.SingletonExtent], tensor.FirstOffset),
				parts[tensor.PairedExtent], tensor.FirstOffset)
		}
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, tensor.FirstOffset, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(tensor.PairedExtent*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		attention := r.attention.graph(builder, q, k, v)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_out.weight"), attention), prefix+"attn_out.bias")
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), prefix+"ffn_up.bias")
		up = builder.GELUTanhExact(up)
		down := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight(visionPostNormWeightTensor), weight(visionPostNormBiasTensor), r.spec.LayerNormEpsilon)
	}
	hidden = builder.WeightedRMSNorm(hidden, weight("mm.pre_norm.weight"), r.spec.LayerNormEpsilon)
	mergedH, mergedW := input.GridH/r.spec.MergeSize, input.GridW/r.spec.MergeSize
	merged := mergePlan.graph(builder, hidden)
	projected := builder.Add(builder.MulMat(conv0Input, merged), weight("mm.0.bias"))
	projected = builder.GELUTanhExact(projected)
	conv2 := builder.Reshape(weight("mm.2.weight"), uint64(r.spec.ConvIntermediate), uint64(r.spec.ProjectorInput))
	projected = builder.Add(builder.MulMat(conv2, projected), weight("mm.2.bias"))
	newline := builder.Reshape(weight(visionImageNewlineTensor), uint64(r.spec.ProjectorInput), tensor.SingletonExtent)
	var content *tensor.Tensor
	for y := range mergedH {
		row := builder.FlatSlice(projected, uint64(y*mergedW*r.spec.ProjectorInput), uint64(r.spec.ProjectorInput), uint64(mergedW))
		row = builder.Concat(row, newline, tensor.SingletonExtent)
		if content == nil {
			content = row
		} else {
			content = builder.Concat(content, row, tensor.SingletonExtent)
		}
	}
	content = builder.Add(builder.MulMat(weight(multimodalProjectionWeight), content), weight(multimodalProjectionBias))
	begin := builder.Reshape(weight("mm.image_begin"), uint64(r.spec.OutputHidden), tensor.SingletonExtent)
	end := builder.Reshape(weight("mm.image_end"), uint64(r.spec.OutputHidden), tensor.SingletonExtent)
	output := builder.Concat(builder.Concat(begin, content, tensor.SingletonExtent), end, tensor.SingletonExtent)
	output = builder.WeightedRMSNorm(output, weight("mm.post_norm.weight"), r.spec.LayerNormEpsilon)
	results, err := graph.execute(output)
	if err != nil {
		return gridOutput{}, fmt.Errorf("projector: execute Hunyuan-VL graph: %w", err)
	}
	return gridOutput{Embeddings: results[output], GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}

func pixelsValue(node *tensor.Tensor, data []float32) reference.Value {
	return reference.Value{Shape: node.Shape, Data: data}
}

func biasedSpatialPositionGraph(
	builder *tensor.Builder,
	hidden, table *tensor.Tensor,
	gridH, gridW, tableSide int,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	rows := gridH * gridW
	sx := (float64(gridW) + interpolationExtentBias) / float64(tableSide)
	sy := (float64(gridH) + interpolationExtentBias) / float64(tableSide)
	indexes := [tensor.MaxDimensions][]uint32{}
	weights := [tensor.MaxDimensions][]float32{}
	for corner := range indexes {
		indexes[corner] = make([]uint32, rows)
		weights[corner] = make([]float32, rows)
	}
	for y := range gridH {
		fy := (float64(y)+media.RasterSampleCenter)/sy - media.RasterSampleCenter
		y0 := max(tensor.FirstOffset, min(tableSide-tensor.SingletonExtent, int(math.Floor(fy))))
		y1 := max(tensor.FirstOffset, min(tableSide-tensor.SingletonExtent, int(math.Floor(fy))+tensor.SingletonExtent))
		wy := min(float64(tensor.SingletonExtent), max(float64(tensor.FirstOffset), fy-float64(y0)))
		for x := range gridW {
			fx := (float64(x)+media.RasterSampleCenter)/sx - media.RasterSampleCenter
			x0 := max(tensor.FirstOffset, min(tableSide-tensor.SingletonExtent, int(math.Floor(fx))))
			x1 := max(tensor.FirstOffset, min(tableSide-tensor.SingletonExtent, int(math.Floor(fx))+tensor.SingletonExtent))
			wx := min(float64(tensor.SingletonExtent), max(float64(tensor.FirstOffset), fx-float64(x0)))
			row := y*gridW + x
			cornerIndexes := [tensor.MaxDimensions]int{y0*tableSide + x0, y0*tableSide + x1, y1*tableSide + x0, y1*tableSide + x1}
			unit := float64(tensor.SingletonExtent)
			cornerWeights := [tensor.MaxDimensions]float64{(unit - wy) * (unit - wx), (unit - wy) * wx, wy * (unit - wx), wy * wx}
			for corner := range indexes {
				indexes[corner][row] = uint32(cornerIndexes[corner])
				weights[corner][row] = float32(cornerWeights[corner])
			}
		}
	}
	var position *tensor.Tensor
	for corner := range indexes {
		factor := builder.Input(fmt.Sprintf("interpolated_position_weight.%d", corner), dtype.F32,
			tensor.MustShape(tensor.SingletonExtent, uint64(rows)))
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
