package projector

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func (r *Llama4VisionRunner) encodeTileCUDA(ctx context.Context, input Llama4VisionTile) (reference.Value, error) {
	patchRows := input.GridH * input.GridW
	patchWidth := 3 * r.spec.PatchSize * r.spec.PatchSize
	if patchRows <= 0 || input.GridH != input.GridW || input.GridH%r.spec.MergeSize != 0 || len(input.PixelValues) != patchRows*patchWidth {
		return reference.Value{}, errors.New("projector: Llama-4 CUDA tile shape is inconsistent")
	}
	mergePlan, err := newPixelMergePlan(input.GridH, input.GridW, r.spec.MergeSize)
	if err != nil {
		return reference.Value{}, err
	}
	rows := patchRows + 1
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(patchRows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[pixels] = pixelsValue(pixels, input.PixelValues)
	weight := graph.weight
	patch := builder.Reshape(weight("v.patch_embd.weight"), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := graph.addOptionalBias(builder.MulMat(patch, pixels), "v.patch_embd.bias")
	class := builder.Reshape(weight("v.class_embd"), uint64(r.spec.Hidden), 1)
	hidden = builder.Concat(hidden, class, 1)
	hidden = builder.Add(hidden, weight("v.position_embd.weight"))
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.pre_ln.weight"), weight("v.pre_ln.bias"), r.spec.LayerNormEpsilon)
	}
	positionsW, positionsH := make([]uint32, rows), make([]uint32, rows)
	for index := 0; index < patchRows; index++ {
		positionsW[index] = uint32(index%input.GridW + 1)
		positionsH[index] = uint32(index/input.GridW + 1)
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
		q = llama4VisionRoPEGraph(builder, q, positionsW, positionsH, r.spec.RopeTheta)
		k = llama4VisionRoPEGraph(builder, k, positionsW, positionsH, r.spec.RopeTheta)
		attention := mustVisionAttentionPlan(r.spec.Hidden, r.spec.Heads, false).graph(builder, q, k, v)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := graph.addOptionalBias(builder.MulMat(weight(prefix+"attn_out.weight"), attention), prefix+"attn_out.bias")
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), prefix+"ffn_up.bias")
		up = r.llama4VisionActivationGraph(builder, up, hostFeeds)
		down := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.post_ln.weight"), weight("v.post_ln.bias"), r.spec.LayerNormEpsilon)
	}
	hidden = builder.FlatSlice(hidden, 0, uint64(r.spec.Hidden), uint64(patchRows))
	merged := mergePlan.graph(builder, hidden)
	adapted := builder.MulMat(weight("mm.model.mlp.1.weight"), merged)
	adapted = qwen3VLGELUTanh(builder, adapted, hostFeeds)
	adapted = builder.MulMat(weight("mm.model.mlp.2.weight"), adapted)
	adapted = qwen3VLGELUTanh(builder, adapted, hostFeeds)
	output := builder.MulMat(weight("mm.model.fc.weight"), adapted)
	results, err := graph.execute(output)
	if err != nil {
		return reference.Value{}, fmt.Errorf("projector: execute Llama-4 CUDA graph: %w", err)
	}
	return results[output], nil
}

func llama4VisionRoPEGraph(
	builder *tensor.Builder,
	input *tensor.Tensor,
	positionsW, positionsH []uint32,
	theta float32,
) *tensor.Tensor {
	headWidth := input.Shape.Dims[0]
	half, heads, rows := headWidth/2, input.Shape.Dims[1], input.Shape.Dims[2]
	w := builder.Reshape(builder.GroupSlice(input, 0, half, 1, half), half, heads, rows)
	h := builder.Reshape(builder.GroupSlice(input, half, half, 1, half), half, heads, rows)
	w = builder.RoPENormal(w, positionsW, uint32(half), theta)
	h = builder.RoPENormal(h, positionsH, uint32(half), theta)
	return builder.Concat(w, h, 0)
}

func (r *Llama4VisionRunner) llama4VisionActivationGraph(
	builder *tensor.Builder,
	input *tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	switch r.spec.Activation {
	case llama4GELU:
		return qwen3VLGELUTanh(builder, input, hostFeeds)
	case llama4SiLU:
		return builder.SiLU(input)
	default:
		return builder.Multiply(input, builder.Sigmoid(builder.Scale(input, 1.702)))
	}
}
