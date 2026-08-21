package projector

import (
	"context"
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *Qwen3VLRunner) encodeGraph(ctx context.Context, input Qwen3VLImage) (Qwen3VLOutput, error) {
	rows := input.GridT * input.GridH * input.GridW
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	pixels0, pixels1, temporalWidth, err := splitTemporalPatchPairs(input.PixelValues, rows, patchArea)
	if err != nil {
		return Qwen3VLOutput{}, err
	}
	builder := tensor.NewBuilder()
	input0 := builder.Input(visionInputTensor+".0", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	input1 := builder.Input(visionInputTensor+".1", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	weight := graph.weight
	patch0 := builder.Reshape(weight(visionPatchWeightTensor), uint64(temporalWidth), uint64(r.spec.Hidden))
	patch1 := builder.Reshape(weight(visionPatchWeightTensor1), uint64(temporalWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.Add(builder.MulMat(patch0, input0), builder.MulMat(patch1, input1)), weight(visionPatchBiasTensor))
	rowOrder, columnOrder := mergedGrid(input.GridH, input.GridW, r.spec.MergeSize)
	hostFeeds := graph.hostFeeds
	hostFeeds[input0] = reference.Value{Shape: input0.Shape, Data: pixels0}
	hostFeeds[input1] = reference.Value{Shape: input1.Shape, Data: pixels1}
	tableSide := r.spec.ImageSize / r.spec.PatchSize
	hidden = spatialPositionGraph(
		builder, hidden, weight(visionPositionWeightTensor), rows, input.GridH*input.GridW,
		input.GridH, input.GridW, tableSide, rowOrder, columnOrder, hostFeeds,
	)
	positionsY, positionsX, spatial := repeatCoordinates(rowOrder, columnOrder, rows)
	var deepstack []*tensor.Tensor
	mergeFactor := r.spec.MergeSize * r.spec.MergeSize
	mergedRows := rows / mergeFactor
	mergedWidth := r.spec.Hidden * mergeFactor
	for layer := range r.spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.AffineLayerNorm(hidden, weight(prefix+"ln1.weight"), weight(prefix+"ln1.bias"), r.spec.LayerNormEpsilon)
		qkv := builder.Add(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), weight(prefix+"attn_qkv.bias"))
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, tensor.FirstOffset, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(tensor.PairedExtent*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		q = interleavedVisionRoPE(builder, q, positionsY, positionsX, r.spec.RopeFrequency)
		k = interleavedVisionRoPE(builder, k, positionsY, positionsX, r.spec.RopeFrequency)
		var attention *tensor.Tensor
		for temporal := range input.GridT {
			offset := uint64(temporal * spatial * r.spec.Hidden)
			shape := []uint64{headWidth, uint64(r.spec.Heads), uint64(spatial)}
			part := builder.AttentionWithOptions(
				builder.FlatSlice(q, offset, shape...), builder.FlatSlice(k, offset, shape...),
				builder.FlatSlice(v, offset, shape...), tensor.AttentionOptions{Scale: float32(tensor.SingletonExtent) / float32(math.Sqrt(float64(headWidth))), Causal: false})

			if attention == nil {
				attention = part
			} else {
				attention = builder.Concat(attention, part, tensor.PairedExtent)
			}
		}
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := builder.Add(builder.MulMat(weight(prefix+"attn_out.weight"), attention), weight(prefix+"attn_out.bias"))
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := builder.Add(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), weight(prefix+"ffn_up.bias"))
		up = builder.GELUTanhExact(up)
		down := builder.Add(builder.MulMat(weight(prefix+"ffn_down.weight"), up), weight(prefix+"ffn_down.bias"))
		hidden = builder.Add(hidden, down)
		if len(r.spec.DeepstackLayers) > layer && r.spec.DeepstackLayers[layer] {
			prefix = fmt.Sprintf("v.deepstack.%d.", layer)
			merged := builder.Reshape(hidden, uint64(mergedWidth), uint64(mergedRows))
			norm = builder.AffineLayerNorm(merged, weight(prefix+"norm.weight"), weight(prefix+"norm.bias"), r.spec.LayerNormEpsilon)
			fc1 := builder.Add(builder.MulMat(weight(prefix+"fc1.weight"), norm), weight(prefix+"fc1.bias"))
			fc1 = builder.GELUTanhExact(fc1)
			deepstack = append(deepstack, builder.Add(builder.MulMat(weight(prefix+"fc2.weight"), fc1), weight(prefix+"fc2.bias")))
		}
	}
	normalized := builder.AffineLayerNorm(hidden, weight(visionPostNormWeightTensor), weight(visionPostNormBiasTensor), r.spec.LayerNormEpsilon)
	merged := builder.Reshape(normalized, uint64(mergedWidth), uint64(mergedRows))
	fc1 := builder.Add(builder.MulMat(weight(projectionFirstWeightTensor), merged), weight(projectionFirstBiasTensor))
	fc1 = builder.GELUTanhExact(fc1)
	output := builder.Add(builder.MulMat(weight(projectionSecondWeightTensor), fc1), weight(projectionSecondBiasTensor))
	targets := append([]*tensor.Tensor{output}, deepstack...)
	results, err := graph.execute(targets...)
	if err != nil {
		return Qwen3VLOutput{}, fmt.Errorf("projector: execute Qwen3-VL graph: %w", err)
	}
	streams := make([]reference.Value, len(deepstack))
	for index, target := range deepstack {
		streams[index] = results[target]
	}
	return Qwen3VLOutput{
		Embeddings: results[output], DeepstackEmbeddings: streams,
		GridT: input.GridT, GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize,
	}, nil
}

func interleavedVisionRoPE(
	builder *tensor.Builder,
	input *tensor.Tensor,
	positionsY, positionsX []uint32,
	frequencyBase float32,
) *tensor.Tensor {
	headWidth := input.Shape.Dims[tensor.FirstOffset]
	quarter := headWidth / (tensor.PairedExtent * tensor.PairedExtent)
	heads := input.Shape.Dims[tensor.SingletonExtent]
	rows := input.Shape.Dims[tensor.PairedExtent]
	axisWidth := headWidth / tensor.PairedExtent
	y := builder.Reshape(
		builder.GroupSlice(input, tensor.FirstOffset, quarter, tensor.PairedExtent, axisWidth),
		axisWidth, heads, rows,
	)
	x := builder.Reshape(
		builder.GroupSlice(input, quarter, quarter, tensor.PairedExtent, axisWidth),
		axisWidth, heads, rows,
	)
	y = builder.RoPEWithOptions(y, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positionsY, RotaryDimensions: uint32(axisWidth), FrequencyBase: frequencyBase, FrequencyScale: tensor.SingletonExtent})
	x = builder.RoPEWithOptions(x, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positionsX, RotaryDimensions: uint32(axisWidth), FrequencyBase: frequencyBase, FrequencyScale: tensor.SingletonExtent})
	split := func(value *tensor.Tensor, offset uint64) *tensor.Tensor {
		return builder.Reshape(builder.GroupSlice(value, offset, quarter, tensor.SingletonExtent, quarter), quarter, heads, rows)
	}
	first := builder.Concat(split(y, tensor.FirstOffset), split(x, tensor.FirstOffset), tensor.FirstOffset)
	second := builder.Concat(split(y, quarter), split(x, quarter), tensor.FirstOffset)
	return builder.Concat(first, second, tensor.FirstOffset)
}
