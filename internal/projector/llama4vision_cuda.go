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

func (r *Llama4VisionRunner) encodeTileCUDA(ctx context.Context, input Llama4VisionTile) (reference.Value, error) {
	patchRows := input.GridH * input.GridW
	patchWidth := 3 * r.spec.PatchSize * r.spec.PatchSize
	if patchRows <= 0 || input.GridH != input.GridW || input.GridH%r.spec.MergeSize != 0 || len(input.PixelValues) != patchRows*patchWidth {
		return reference.Value{}, errors.New("projector: Llama-4 CUDA tile shape is inconsistent")
	}
	rows := patchRows + 1
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(patchRows)))
	hostFeeds := map[*tensor.Tensor]reference.Value{pixels: pixelsValue(pixels, input.PixelValues)}
	binding := r.cuda.bindWeights(builder)
	weight := binding.weight
	addBias := func(value *tensor.Tensor, name string) *tensor.Tensor {
		if !hasTensor(r.file, name) {
			return value
		}
		return builder.Add(value, weight(name))
	}
	patch := builder.Reshape(weight("v.patch_embd.weight"), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := addBias(builder.MulMat(patch, pixels), "v.patch_embd.bias")
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
			qkv = addBias(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), prefix+"attn_qkv.bias")
		} else {
			parts := make([]*tensor.Tensor, 3)
			for index, part := range []string{"q", "k", "v"} {
				parts[index] = addBias(builder.MulMat(weight(prefix+"attn_"+part+".weight"), norm), prefix+"attn_"+part+".bias")
			}
			qkv = builder.Concat(builder.Concat(parts[0], parts[1], 0), parts[2], 0)
		}
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, 0, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(2*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		q = llama4VisionRoPEGraph(builder, q, positionsW, positionsH, r.spec.RopeTheta)
		k = llama4VisionRoPEGraph(builder, k, positionsW, positionsH, r.spec.RopeTheta)
		attention := builder.Attention(q, k, v, float32(1/math.Sqrt(float64(headWidth))), false)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := addBias(builder.MulMat(weight(prefix+"attn_out.weight"), attention), prefix+"attn_out.bias")
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := addBias(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), prefix+"ffn_up.bias")
		up = r.llama4VisionActivationGraph(builder, up, hostFeeds)
		down := addBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.post_ln.weight"), weight("v.post_ln.bias"), r.spec.LayerNormEpsilon)
	}
	hidden = builder.FlatSlice(hidden, 0, uint64(r.spec.Hidden), uint64(patchRows))
	mergedH, mergedW := input.GridH/r.spec.MergeSize, input.GridW/r.spec.MergeSize
	mergeFactor := r.spec.MergeSize * r.spec.MergeSize
	indexSets := make([][]uint32, mergeFactor)
	for index := range indexSets {
		indexSets[index] = make([]uint32, 0, mergedH*mergedW)
	}
	for blockY := 0; blockY < mergedH; blockY++ {
		for blockX := 0; blockX < mergedW; blockX++ {
			for y := 0; y < r.spec.MergeSize; y++ {
				for x := 0; x < r.spec.MergeSize; x++ {
					offset := y*r.spec.MergeSize + x
					indexSets[offset] = append(indexSets[offset], uint32((blockY*r.spec.MergeSize+y)*input.GridW+blockX*r.spec.MergeSize+x))
				}
			}
		}
	}
	merged := builder.GetRows(hidden, indexSets[0])
	for offset := 1; offset < len(indexSets); offset++ {
		merged = builder.Concat(merged, builder.GetRows(hidden, indexSets[offset]), 0)
	}
	adapted := builder.MulMat(weight("mm.model.mlp.1.weight"), merged)
	adapted = qwen3VLGELUTanh(builder, adapted, hostFeeds)
	adapted = builder.MulMat(weight("mm.model.mlp.2.weight"), adapted)
	adapted = qwen3VLGELUTanh(builder, adapted, hostFeeds)
	output := builder.MulMat(weight("mm.model.fc.weight"), adapted)
	deviceFeeds, err := binding.result()
	if err != nil {
		return reference.Value{}, fmt.Errorf("projector: build Llama-4 CUDA graph: %w", err)
	}
	if err := builder.Err(); err != nil {
		return reference.Value{}, fmt.Errorf("projector: build Llama-4 CUDA graph: %w", err)
	}
	results, err := r.cuda.executor.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{output}, hostFeeds, deviceFeeds)
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
