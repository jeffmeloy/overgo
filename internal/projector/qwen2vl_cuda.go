package projector

import (
	"context"
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *Qwen2VLRunner) encodeGraph(ctx context.Context, input Qwen2VLImage) (Qwen2VLOutput, error) {
	rows := input.GridT * input.GridH * input.GridW
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	pixels0, pixels1, temporalWidth, err := splitTemporalPatchPairs(input.PixelValues, rows, patchArea)
	if err != nil {
		return Qwen2VLOutput{}, err
	}
	builder := tensor.NewBuilder()
	input0 := builder.Input("pixel_values.0", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	input1 := builder.Input("pixel_values.1", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	weight := graph.weight
	patch0 := builder.Reshape(weight(visionPatchWeightTensor), uint64(temporalWidth), uint64(r.spec.Hidden))
	patch1 := builder.Reshape(weight(visionPatchWeightTensor1), uint64(temporalWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch0, input0), builder.MulMat(patch1, input1))
	graph.hostFeeds[input0] = reference.Value{Shape: input0.Shape, Data: pixels0}
	graph.hostFeeds[input1] = reference.Value{Shape: input1.Shape, Data: pixels1}
	hostFeeds := graph.hostFeeds
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight(visionPreNormWeightTensor), weight(visionPreNormBiasTensor), r.spec.LayerNormEpsilon)
	}
	rowOrder, columnOrder := mergedGrid(input.GridH, input.GridW, r.spec.MergeSize)
	positionsY := make([]uint32, rows)
	positionsX := make([]uint32, rows)
	spatial := input.GridH * input.GridW
	for row := 0; row < rows; row++ {
		positionsY[row] = uint32(rowOrder[row%spatial])
		positionsX[row] = uint32(columnOrder[row%spatial])
	}
	headWidth := uint64(r.spec.Hidden / r.spec.Heads)
	for layer := 0; layer < r.spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.AffineLayerNorm(hidden, weight(prefix+"ln1.weight"), weight(prefix+"ln1.bias"), r.spec.LayerNormEpsilon)
		project := func(name string) *tensor.Tensor {
			return builder.Add(builder.MulMat(weight(prefix+name+".weight"), norm), weight(prefix+name+".bias"))
		}
		q := builder.Reshape(project("attn_q"), headWidth, uint64(r.spec.Heads), uint64(rows))
		k := builder.Reshape(project("attn_k"), headWidth, uint64(r.spec.Heads), uint64(rows))
		v := builder.Reshape(project("attn_v"), headWidth, uint64(r.spec.Heads), uint64(rows))
		q = interleavedVisionRoPE(builder, q, positionsY, positionsX, r.spec.RopeFrequency)
		k = interleavedVisionRoPE(builder, k, positionsY, positionsX, r.spec.RopeFrequency)
		var attention *tensor.Tensor
		for temporal := 0; temporal < input.GridT; temporal++ {
			offset := uint64(temporal * spatial * r.spec.Hidden)
			shape := []uint64{headWidth, uint64(r.spec.Heads), uint64(spatial)}
			part := builder.AttentionWithOptions(
				builder.FlatSlice(q, offset, shape...), builder.FlatSlice(k, offset, shape...),
				builder.FlatSlice(v, offset, shape...), tensor.AttentionOptions{Scale: float32(1 / math.Sqrt(float64(headWidth))), Causal: false})

			if attention == nil {
				attention = part
			} else {
				attention = builder.Concat(attention, part, 2)
			}
		}
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := builder.Add(builder.MulMat(weight(prefix+"attn_out.weight"), attention), weight(prefix+"attn_out.bias"))
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := builder.Add(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), weight(prefix+"ffn_up.bias"))
		up = qwen3VLGELUTanh(builder, up, hostFeeds)
		down := builder.Add(builder.MulMat(weight(prefix+"ffn_down.weight"), up), weight(prefix+"ffn_down.bias"))
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight(visionPostNormWeightTensor), weight(visionPostNormBiasTensor), r.spec.LayerNormEpsilon)
	}
	mergedRows := rows / (r.spec.MergeSize * r.spec.MergeSize)
	merged := builder.Reshape(hidden, uint64(r.spec.Hidden*r.spec.MergeSize*r.spec.MergeSize), uint64(mergedRows))
	fc1 := builder.Add(builder.MulMat(weight(projectionFirstWeightTensor), merged), weight(projectionFirstBiasTensor))
	fc1 = qwen3VLGELUTanh(builder, fc1, hostFeeds)
	output := builder.Add(builder.MulMat(weight(projectionSecondWeightTensor), fc1), weight(projectionSecondBiasTensor))
	results, err := graph.execute(output)
	if err != nil {
		return Qwen2VLOutput{}, fmt.Errorf("projector: execute Qwen2-VL graph: %w", err)
	}
	return Qwen2VLOutput{Embeddings: results[output], GridT: input.GridT, GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}
