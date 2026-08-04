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

func (r *MiMoVLRunner) encodeGraph(ctx context.Context, input MiMoVLInput) (MiMoVLOutput, error) {
	rows := input.GridH * input.GridW
	if rows <= 0 || input.GridH%r.spec.MergeSize != 0 || input.GridW%r.spec.MergeSize != 0 {
		return MiMoVLOutput{}, errors.New("projector: MiMo-VL input geometry is inconsistent")
	}
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	pixels0, pixels1, temporalWidth, err := splitTemporalPatchPairs(input.PixelValues, rows, patchArea)
	if err != nil {
		return MiMoVLOutput{}, err
	}
	builder := tensor.NewBuilder()
	input0 := builder.Input("pixel_values.0", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	input1 := builder.Input("pixel_values.1", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[input0] = reference.Value{Shape: input0.Shape, Data: pixels0}
	hostFeeds[input1] = reference.Value{Shape: input1.Shape, Data: pixels1}
	weight := graph.weight
	patch0 := builder.Reshape(weight("v.patch_embd.weight"), uint64(temporalWidth), uint64(r.spec.Hidden))
	patch1 := builder.Reshape(weight("v.patch_embd.weight.1"), uint64(temporalWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch0, input0), builder.MulMat(patch1, input1))
	rowPositions, columnPositions := mergedGrid(input.GridH, input.GridW, r.spec.MergeSize)
	positionsH, positionsW := intsToUint32(rowPositions), intsToUint32(columnPositions)
	columnOrder := mimoVLColumnOrder(input.GridH/r.spec.MergeSize, input.GridW/r.spec.MergeSize, r.spec.MergeSize)
	inverseColumnOrder := inversePermutation(columnOrder)
	previousMode := -1
	qWidth := r.spec.Heads * r.spec.HeadDim
	kvWidth := r.spec.KVHeads * r.spec.HeadDim
	for layer, mode := range r.spec.WindowModes {
		if mode == 1 && previousMode != 1 {
			hidden = builder.GetRows(hidden, intsToUint32(columnOrder))
			positionsH = intsToUint32(reorderInts(rowPositions, columnOrder))
			positionsW = intsToUint32(reorderInts(columnPositions, columnOrder))
		} else if mode != 1 && previousMode == 1 {
			hidden = builder.GetRows(hidden, intsToUint32(inverseColumnOrder))
			positionsH, positionsW = intsToUint32(rowPositions), intsToUint32(columnPositions)
		}
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := graph.weightedRMSNorm(hidden, prefix+"ln1", r.spec.LayerNormEpsilon)
		qkv := builder.Add(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), weight(prefix+"attn_qkv.bias"))
		headDim := uint64(r.spec.HeadDim)
		q := builder.GroupSlice(qkv, 0, headDim, uint64(r.spec.Heads), headDim)
		k := builder.GroupSlice(qkv, uint64(qWidth), headDim, uint64(r.spec.KVHeads), headDim)
		v := builder.GroupSlice(qkv, uint64(qWidth+kvWidth), headDim, uint64(r.spec.KVHeads), headDim)
		q = qwen3VLVisionRoPE(builder, q, positionsH, positionsW)
		k = qwen3VLVisionRoPE(builder, k, positionsH, positionsW)
		scale := float32(1 / math.Sqrt(float64(r.spec.HeadDim)))
		var attention *tensor.Tensor
		if mode == -1 {
			attention = builder.Attention(q, k, v, scale, false)
		} else {
			attention = builder.AttentionSymmetricWindowWithSinks(
				q, k, v, weight(prefix+"attn_sinks"), scale, uint32(2*r.spec.WindowSize),
			)
		}
		attention = builder.Reshape(attention, uint64(qWidth), uint64(rows))
		projected := builder.MulMat(weight(prefix+"attn_out.weight"), attention)
		hidden = builder.Add(hidden, graph.addOptionalBias(projected, prefix+"attn_out.bias"))
		norm = graph.weightedRMSNorm(hidden, prefix+"ln2", r.spec.LayerNormEpsilon)
		up := builder.Add(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), weight(prefix+"ffn_up.bias"))
		gate := builder.Add(builder.MulMat(weight(prefix+"ffn_gate.weight"), norm), weight(prefix+"ffn_gate.bias"))
		activated := builder.Multiply(builder.SiLU(gate), up)
		down := builder.Add(builder.MulMat(weight(prefix+"ffn_down.weight"), activated), weight(prefix+"ffn_down.bias"))
		hidden = builder.Add(hidden, down)
		previousMode = mode
	}
	if previousMode == 1 {
		hidden = builder.GetRows(hidden, intsToUint32(inverseColumnOrder))
	}
	normalized := builder.Multiply(builder.LayerNorm(hidden, mimoVLPostNormEpsilon), weight("v.post_ln.weight"))
	normalized = graph.addOptionalBias(normalized, "v.post_ln.bias")
	mergedRows := rows / (r.spec.MergeSize * r.spec.MergeSize)
	merged := builder.Reshape(normalized, uint64(r.spec.Hidden*4), uint64(mergedRows))
	fc1 := builder.MulMat(weight("mm.0.weight"), merged)
	fc1 = qwen3VLGELUTanh(builder, graph.addOptionalBias(fc1, "mm.0.bias"), hostFeeds)
	output := builder.MulMat(weight("mm.2.weight"), fc1)
	output = graph.addOptionalBias(output, "mm.2.bias")
	results, err := graph.execute(output)
	if err != nil {
		return MiMoVLOutput{}, fmt.Errorf("projector: execute MiMo-VL graph: %w", err)
	}
	return MiMoVLOutput{
		Embeddings: results[output], GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize,
	}, nil
}

func intsToUint32(values []int) []uint32 {
	output := make([]uint32, len(values))
	for index, value := range values {
		output[index] = uint32(value)
	}
	return output
}
