package projector

import (
	"context"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func (r *CogVLMVisionRunner) encodeGraph(ctx context.Context, pixelsData []float32) (reference.Value, error) {
	grid := r.spec.ImageSize / r.spec.PatchSize
	patchRows, patchWidth := grid*grid, 3*r.spec.PatchSize*r.spec.PatchSize
	rows := patchRows + 1
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(patchRows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[pixels] = pixelsValue(pixels, pixelsData)
	weight := graph.weight
	hostFeeds := graph.hostFeeds
	patch := builder.Reshape(weight("v.patch_embd.weight"), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := graph.addOptionalBias(builder.MulMat(patch, pixels), "v.patch_embd.bias")
	hidden = builder.Concat(hidden, builder.Reshape(weight("v.class_embd"), uint64(r.spec.Hidden), 1), 1)
	hidden = builder.Add(hidden, weight("v.position_embd.weight"))
	for layer := 0; layer < r.spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		qkv := builder.Add(builder.MulMat(weight(prefix+"attn_qkv.weight"), hidden), weight(prefix+"attn_qkv.bias"))
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, 0, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(2*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		attention := builder.Attention(q, k, v, float32(1/math.Sqrt(float64(headWidth))), false)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		attention = builder.Add(builder.MulMat(weight(prefix+"attn_out.weight"), attention), weight(prefix+"attn_out.bias"))
		attention = builder.AffineLayerNorm(attention, weight(prefix+"ln1.weight"), weight(prefix+"ln1.bias"), r.spec.LayerNormEpsilon)
		hidden = builder.Add(hidden, attention)
		up := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_up.weight"), hidden), prefix+"ffn_up.bias")
		if r.spec.GatedFFN[layer] {
			gate := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_gate.weight"), hidden), prefix+"ffn_gate.bias")
			up = builder.Multiply(up, qwen3VLGELUTanh(builder, gate, hostFeeds))
		} else {
			up = qwen3VLGELUTanh(builder, up, hostFeeds)
		}
		ffn := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		ffn = builder.AffineLayerNorm(ffn, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		hidden = builder.Add(hidden, ffn)
	}
	hidden = builder.FlatSlice(hidden, 0, uint64(r.spec.Hidden), uint64(patchRows))
	hidden = builder.MulMat(weight("mm.model.fc.weight"), hidden)
	hidden = builder.AffineLayerNorm(
		hidden, weight("mm.post_fc_norm.weight"), weight("mm.post_fc_norm.bias"), cogVLMAdapterNormEpsilon,
	)
	hidden = qwen3VLGELUTanh(builder, hidden, hostFeeds)
	up := builder.MulMat(weight("mm.up.weight"), hidden)
	gate := builder.SiLU(builder.MulMat(weight("mm.gate.weight"), hidden))
	hidden = builder.MulMat(weight("mm.down.weight"), builder.Multiply(gate, up))
	boi := builder.Reshape(weight("v.boi"), uint64(r.spec.OutputHidden), 1)
	eoi := builder.Reshape(weight("v.eoi"), uint64(r.spec.OutputHidden), 1)
	output := builder.Concat(builder.Concat(boi, hidden, 1), eoi, 1)
	results, err := graph.execute(output)
	if err != nil {
		return reference.Value{}, fmt.Errorf("projector: execute CogVLM graph: %w", err)
	}
	return results[output], nil
}
