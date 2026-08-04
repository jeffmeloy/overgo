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

func (r *Qwen2VLRunner) encodeGraph(ctx context.Context, input Qwen2VLImage) (Qwen2VLOutput, error) {
	rows := input.GridT * input.GridH * input.GridW
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	temporalWidth := 3 * patchArea
	if len(input.PixelValues) != rows*temporalWidth*2 {
		return Qwen2VLOutput{}, errors.New("projector: Qwen2-VL input shape is inconsistent")
	}
	pixels0 := make([]float32, rows*temporalWidth)
	pixels1 := make([]float32, rows*temporalWidth)
	for row := 0; row < rows; row++ {
		source := input.PixelValues[row*temporalWidth*2:]
		for color := 0; color < 3; color++ {
			copy(pixels0[row*temporalWidth+color*patchArea:], source[color*2*patchArea:color*2*patchArea+patchArea])
			copy(pixels1[row*temporalWidth+color*patchArea:], source[color*2*patchArea+patchArea:(color+1)*2*patchArea])
		}
	}
	builder := tensor.NewBuilder()
	input0 := builder.Input("pixel_values.0", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	input1 := builder.Input("pixel_values.1", dtype.F32, tensor.MustShape(uint64(temporalWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	weight := graph.weight
	patch0 := builder.Reshape(weight("v.patch_embd.weight"), uint64(temporalWidth), uint64(r.spec.Hidden))
	patch1 := builder.Reshape(weight("v.patch_embd.weight.1"), uint64(temporalWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch0, input0), builder.MulMat(patch1, input1))
	graph.hostFeeds[input0] = reference.Value{Shape: input0.Shape, Data: pixels0}
	graph.hostFeeds[input1] = reference.Value{Shape: input1.Shape, Data: pixels1}
	hostFeeds := graph.hostFeeds
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.pre_ln.weight"), weight("v.pre_ln.bias"), r.spec.LayerNormEpsilon)
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
		q = qwen3VLVisionRoPE(builder, q, positionsY, positionsX)
		k = qwen3VLVisionRoPE(builder, k, positionsY, positionsX)
		var attention *tensor.Tensor
		for temporal := 0; temporal < input.GridT; temporal++ {
			offset := uint64(temporal * spatial * r.spec.Hidden)
			shape := []uint64{headWidth, uint64(r.spec.Heads), uint64(spatial)}
			part := builder.Attention(
				builder.FlatSlice(q, offset, shape...), builder.FlatSlice(k, offset, shape...),
				builder.FlatSlice(v, offset, shape...), float32(1/math.Sqrt(float64(headWidth))), false,
			)
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
		upName, downName := "ffn_up", "ffn_down"
		if r.spec.LegacyFFNSwapped {
			upName, downName = downName, upName
		}
		up := builder.Add(builder.MulMat(weight(prefix+upName+".weight"), norm), weight(prefix+upName+".bias"))
		up = qwen3VLGELUTanh(builder, up, hostFeeds)
		down := builder.Add(builder.MulMat(weight(prefix+downName+".weight"), up), weight(prefix+downName+".bias"))
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.post_ln.weight"), weight("v.post_ln.bias"), r.spec.LayerNormEpsilon)
	}
	mergedRows := rows / 4
	merged := builder.Reshape(hidden, uint64(r.spec.Hidden*4), uint64(mergedRows))
	fc1 := builder.Add(builder.MulMat(weight("mm.0.weight"), merged), weight("mm.0.bias"))
	fc1 = qwen3VLGELUTanh(builder, fc1, hostFeeds)
	output := builder.Add(builder.MulMat(weight("mm.2.weight"), fc1), weight("mm.2.bias"))
	results, err := graph.execute(output)
	if err != nil {
		return Qwen2VLOutput{}, fmt.Errorf("projector: execute Qwen2-VL graph: %w", err)
	}
	return Qwen2VLOutput{Embeddings: results[output], GridT: input.GridT, GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}
