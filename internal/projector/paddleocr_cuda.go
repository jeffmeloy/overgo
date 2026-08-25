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

func (r *PaddleOCRRunner) encodeGraph(ctx context.Context, input RasterPatchImage) (gridOutput, error) {
	rows, patchWidth, err := validateSpatialPatchStorage(
		len(input.PixelValues), input.GridH, input.GridW, r.spec.PatchSize, media.RGBChannels,
	)
	if err != nil {
		return gridOutput{}, errors.New("projector: PaddleOCR input shape is inconsistent")
	}
	mergePlan, err := newPixelMergePlan(input.GridH, input.GridW, r.spec.MergeSize)
	if err != nil {
		return gridOutput{}, err
	}
	builder := tensor.NewBuilder()
	pixels := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[pixels] = reference.Value{Shape: pixels.Shape, Data: input.PixelValues}
	weight := graph.weight
	patch := builder.Reshape(weight(visionPatchWeightTensor), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := graph.addOptionalBias(builder.MulMat(patch, pixels), visionPatchBiasTensor)
	rowOrder, columnOrder := gridCoordinates(input.GridH, input.GridW)
	tableSide := r.spec.ImageSize / r.spec.PatchSize
	hidden = spatialPositionGraph(
		builder, hidden, weight(visionPositionWeightTensor), rows, rows,
		input.GridH, input.GridW, tableSide, rowOrder, columnOrder, hostFeeds,
	)
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight(visionPreNormWeightTensor), weight(visionPreNormBiasTensor), r.spec.LayerNormEpsilon)
	}
	positionsY, positionsX := intsToUint32(rowOrder), intsToUint32(columnOrder)
	for layer := range r.spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.AffineLayerNorm(hidden, weight(prefix+"ln1.weight"), weight(prefix+"ln1.bias"), r.spec.LayerNormEpsilon)
		var qkv *tensor.Tensor
		if r.spec.FusedQKV[layer] {
			qkv = graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), prefix+"attn_qkv.bias")
		} else {
			parts := make([]*tensor.Tensor, tensor.TripleExtent)
			for index, part := range []string{"q", "k", "v"} {
				parts[index] = graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_"+part+".weight"), norm), prefix+"attn_"+part+".bias")
			}
			qkv = builder.Concat(
				builder.Concat(parts[tensor.FirstOffset], parts[tensor.SingletonExtent], tensor.FirstOffset),
				parts[tensor.PairedExtent], tensor.FirstOffset,
			)
		}
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, tensor.FirstOffset, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(tensor.PairedExtent*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		q = interleavedVisionRoPE(builder, q, positionsY, positionsX, r.spec.RopeFrequency)
		k = interleavedVisionRoPE(builder, k, positionsY, positionsX, r.spec.RopeFrequency)
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
	hidden = builder.AffineLayerNorm(
		hidden, weight("mm.input_norm.weight"), weight("mm.input_norm.bias"), r.spec.ProjectionNormEpsilon,
	)
	merged := mergePlan.graph(builder, hidden)
	fc1 := builder.Add(builder.MulMat(weight("mm.1.weight"), merged), weight("mm.1.bias"))
	fc1 = visionActivationNode(builder, fc1, r.spec.Activation)
	output := builder.Add(builder.MulMat(weight(projectionSecondWeightTensor), fc1), weight(projectionSecondBiasTensor))
	results, err := graph.execute(output)
	if err != nil {
		return gridOutput{}, fmt.Errorf("projector: execute PaddleOCR graph: %w", err)
	}
	return gridOutput{
		Embeddings: results[output], GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize,
	}, nil
}
