package projector

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *Llama4VisionRunner) encodeTileCUDA(ctx context.Context, input Llama4VisionTile) (reference.Value, error) {
	patchRows, patchWidth, err := validateSpatialPatchStorage(
		len(input.PixelValues), input.GridH, input.GridW, r.spec.PatchSize, media.RGBChannels,
	)
	if err != nil || input.GridH != input.GridW {
		return reference.Value{}, errors.New("projector: Llama-4 CUDA tile shape is inconsistent")
	}
	mergePlan, err := newPixelMergePlan(input.GridH, input.GridW, r.spec.MergeSize)
	if err != nil {
		return reference.Value{}, err
	}
	rows := patchRows + tensor.SingletonExtent
	builder := tensor.NewBuilder()
	pixels := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(patchRows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[pixels] = pixelsValue(pixels, input.PixelValues)
	weight := graph.weight
	patch := builder.Reshape(weight(visionPatchWeightTensor), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := graph.addOptionalBias(builder.MulMat(patch, pixels), visionPatchBiasTensor)
	class := builder.Reshape(weight(visionClassEmbeddingTensor), uint64(r.spec.Hidden), tensor.SingletonExtent)
	hidden = builder.Concat(hidden, class, tensor.SingletonExtent)
	hidden = builder.Add(hidden, weight(visionPositionWeightTensor))
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight(visionPreNormWeightTensor), weight(visionPreNormBiasTensor), r.spec.LayerNormEpsilon)
	}
	positionsW, positionsH := make([]uint32, rows), make([]uint32, rows)
	for index := range patchRows {
		positionsW[index] = uint32(index%input.GridW + tensor.SingletonExtent)
		positionsH[index] = uint32(index/input.GridW + tensor.SingletonExtent)
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
		q = spatialRotaryGraph(builder, q, positionsW, positionsH, r.spec.RopeFrequency)
		k = spatialRotaryGraph(builder, k, positionsW, positionsH, r.spec.RopeFrequency)
		attention := r.attention.graph(builder, q, k, v)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_out.weight"), attention), prefix+"attn_out.bias")
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), prefix+"ffn_up.bias")
		up = visionActivationNode(builder, up, r.spec.Activation)
		down := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight(visionPostNormWeightTensor), weight(visionPostNormBiasTensor), r.spec.LayerNormEpsilon)
	}
	hidden = builder.FlatSlice(hidden, tensor.FirstOffset, uint64(r.spec.Hidden), uint64(patchRows))
	merged := mergePlan.graph(builder, hidden)
	adapted := builder.MulMat(weight("mm.model.mlp.1.weight"), merged)
	adapted = builder.GELUTanhExact(adapted)
	adapted = builder.MulMat(weight("mm.model.mlp.2.weight"), adapted)
	adapted = builder.GELUTanhExact(adapted)
	output := builder.MulMat(weight(multimodalProjectionWeight), adapted)
	results, err := graph.execute(output)
	if err != nil {
		return reference.Value{}, fmt.Errorf("projector: execute Llama-4 CUDA graph: %w", err)
	}
	return results[output], nil
}

func spatialRotaryGraph(
	builder *tensor.Builder,
	input *tensor.Tensor,
	positionsW, positionsH []uint32,
	theta float32,
) *tensor.Tensor {
	headWidth := input.Shape.Dims[tensor.FirstOffset]
	half := headWidth / tensor.PairedExtent
	heads := input.Shape.Dims[tensor.SingletonExtent]
	rows := input.Shape.Dims[tensor.PairedExtent]
	w := builder.Reshape(builder.GroupSlice(input, tensor.FirstOffset, half, tensor.SingletonExtent, half), half, heads, rows)
	h := builder.Reshape(builder.GroupSlice(input, half, half, tensor.SingletonExtent, half), half, heads, rows)
	w = builder.RoPEWithOptions(w, tensor.RoPEOptions{Layout: tensor.RoPELayoutNormal, Positions: positionsW, RotaryDimensions: uint32(half), FrequencyBase: theta, FrequencyScale: tensor.SingletonExtent})
	h = builder.RoPEWithOptions(h, tensor.RoPEOptions{Layout: tensor.RoPELayoutNormal, Positions: positionsH, RotaryDimensions: uint32(half), FrequencyBase: theta, FrequencyScale: tensor.SingletonExtent})
	return builder.Concat(w, h, tensor.FirstOffset)
}
