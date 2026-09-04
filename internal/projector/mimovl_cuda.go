package projector

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *MiMoVLRunner) encodeGraph(ctx context.Context, input MiMoVLInput) (MiMoVLOutput, error) {
	mergePlan, err := newPixelMergePlan(input.GridH, input.GridW, r.spec.MergeSize)
	if err != nil {
		return MiMoVLOutput{}, errors.New("projector: MiMo-VL input geometry is inconsistent")
	}
	rows := mergePlan.inputRows
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	pixels0, pixels1, temporalWidth, err := splitTemporalPatchPairs(input.PixelValues, rows, patchArea)
	if err != nil {
		return MiMoVLOutput{}, err
	}
	builder := tensor.NewBuilder()
	input0 := builder.Input(visionInputTensor+".0", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	input1 := builder.Input(visionInputTensor+".1", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[input0] = reference.Value{Shape: input0.Shape, Data: pixels0}
	hostFeeds[input1] = reference.Value{Shape: input1.Shape, Data: pixels1}
	weight := graph.weight
	patch0 := builder.Reshape(weight(visionPatchWeightTensor), uint64(temporalWidth), uint64(r.spec.Hidden))
	patch1 := builder.Reshape(weight(visionPatchWeightTensor1), uint64(temporalWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch0, input0), builder.MulMat(patch1, input1))
	rowPositions, columnPositions := mergedGrid(input.GridH, input.GridW, r.spec.MergeSize)
	basePositionsH, basePositionsW := intsToUint32(rowPositions), intsToUint32(columnPositions)
	positionsH, positionsW := basePositionsH, basePositionsW
	columnOrder := columnMajorPatchOrder(mergePlan.outputHeight, mergePlan.outputWidth, r.spec.MergeSize)
	inverseColumnOrder := inversePermutation(columnOrder)
	columnOrderRows, inverseColumnOrderRows := intsToUint32(columnOrder), intsToUint32(inverseColumnOrder)
	columnPositionsH := intsToUint32(reorderInts(rowPositions, columnOrder))
	columnPositionsW := intsToUint32(reorderInts(columnPositions, columnOrder))
	previousMode := windowModeGlobal
	qWidth := r.spec.Heads * r.spec.HeadDim
	kvWidth := r.spec.KVHeads * r.spec.HeadDim
	for layer, mode := range r.spec.WindowModes {
		if mode == windowModeColumn && previousMode != windowModeColumn {
			hidden = builder.GetRows(hidden, columnOrderRows)
			positionsH, positionsW = columnPositionsH, columnPositionsW
		} else if mode != windowModeColumn && previousMode == windowModeColumn {
			hidden = builder.GetRows(hidden, inverseColumnOrderRows)
			positionsH, positionsW = basePositionsH, basePositionsW
		}
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := graph.weightedRMSNorm(hidden, prefix+"ln1", r.spec.LayerNormEpsilon)
		qkv := builder.Add(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), weight(prefix+"attn_qkv.bias"))
		headDim := uint64(r.spec.HeadDim)
		q := builder.GroupSlice(qkv, tensor.FirstOffset, headDim, uint64(r.spec.Heads), headDim)
		k := builder.GroupSlice(qkv, uint64(qWidth), headDim, uint64(r.spec.KVHeads), headDim)
		v := builder.GroupSlice(qkv, uint64(qWidth+kvWidth), headDim, uint64(r.spec.KVHeads), headDim)
		q = interleavedVisionRoPE(builder, q, positionsH, positionsW, r.spec.RopeFrequency)
		k = interleavedVisionRoPE(builder, k, positionsH, positionsW, r.spec.RopeFrequency)
		scale := float32(tensor.SingletonExtent) / float32(math.Sqrt(float64(r.spec.HeadDim)))
		var attention *tensor.Tensor
		if mode == windowModeGlobal {
			attention = builder.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: scale, Causal: false})
		} else {
			attention = builder.AttentionWithOptions(
				q, k, v, tensor.AttentionOptions{Sinks: weight(prefix + "attn_sinks"), Scale: scale, SymmetricWindow: true, Window: uint32(tensor.PairedExtent * r.spec.WindowSize)})

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
	if previousMode == windowModeColumn {
		hidden = builder.GetRows(hidden, inverseColumnOrderRows)
	}
	normalized := builder.Multiply(builder.LayerNorm(hidden, r.spec.ProjectionNormEpsilon), weight(visionPostNormWeightTensor))
	normalized = graph.addOptionalBias(normalized, visionPostNormBiasTensor)
	merged := builder.Reshape(normalized, uint64(r.spec.Hidden*r.spec.MergeSize*r.spec.MergeSize), uint64(mergePlan.outputRows))
	fc1 := builder.MulMat(weight(projectionFirstWeightTensor), merged)
	fc1 = builder.GELUTanhExact(graph.addOptionalBias(fc1, projectionFirstBiasTensor))
	output := builder.MulMat(weight(projectionSecondWeightTensor), fc1)
	output = graph.addOptionalBias(output, projectionSecondBiasTensor)
	results, err := graph.execute(output)
	if err != nil {
		return MiMoVLOutput{}, fmt.Errorf("projector: execute MiMo-VL graph: %w", err)
	}
	return MiMoVLOutput{
		Embeddings: results[output], GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize,
	}, nil
}

func intsToUint32(values []int) []uint32 { return dtype.ConvertSlice[uint32](values) }
