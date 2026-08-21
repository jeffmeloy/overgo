package projector

import (
	"context"
	"fmt"

	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *CogVLMVisionRunner) encodeGraph(ctx context.Context, pixelsData []float32) (reference.Value, error) {
	grid := r.spec.ImageSize / r.spec.PatchSize
	patchRows, patchWidth := grid*grid, media.RGBChannels*r.spec.PatchSize*r.spec.PatchSize
	rows := patchRows + tensor.SingletonExtent
	builder := tensor.NewBuilder()
	pixels := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(patchRows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[pixels] = pixelsValue(pixels, pixelsData)
	weight := graph.weight
	patch := builder.Reshape(weight(visionPatchWeightTensor), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := graph.addOptionalBias(builder.MulMat(patch, pixels), visionPatchBiasTensor)
	hidden = builder.Concat(hidden, builder.Reshape(weight(visionClassEmbeddingTensor), uint64(r.spec.Hidden), tensor.SingletonExtent), tensor.SingletonExtent)
	hidden = builder.Add(hidden, weight(visionPositionWeightTensor))
	for layer := range r.spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		qkv := builder.Add(builder.MulMat(weight(prefix+"attn_qkv.weight"), hidden), weight(prefix+"attn_qkv.bias"))
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, tensor.FirstOffset, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(tensor.PairedExtent*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		attention := r.attention.graph(builder, q, k, v)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		attention = builder.Add(builder.MulMat(weight(prefix+"attn_out.weight"), attention), weight(prefix+"attn_out.bias"))
		attention = builder.AffineLayerNorm(attention, weight(prefix+"ln1.weight"), weight(prefix+"ln1.bias"), r.spec.LayerNormEpsilon)
		hidden = builder.Add(hidden, attention)
		up := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_up.weight"), hidden), prefix+"ffn_up.bias")
		if r.spec.GatedFFN[layer] {
			gate := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_gate.weight"), hidden), prefix+"ffn_gate.bias")
			up = builder.Multiply(up, builder.GELUTanhExact(gate))
		} else {
			up = builder.GELUTanhExact(up)
		}
		ffn := graph.addOptionalBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		ffn = builder.AffineLayerNorm(ffn, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		hidden = builder.Add(hidden, ffn)
	}
	hidden = builder.FlatSlice(hidden, tensor.FirstOffset, uint64(r.spec.Hidden), uint64(patchRows))
	hidden = builder.MulMat(weight(multimodalProjectionWeight), hidden)
	hidden = builder.AffineLayerNorm(
		hidden, weight("mm.post_fc_norm.weight"), weight("mm.post_fc_norm.bias"), r.spec.ProjectionNormEpsilon,
	)
	hidden = builder.GELUTanhExact(hidden)
	up := builder.MulMat(weight("mm.up.weight"), hidden)
	gate := builder.SiLU(builder.MulMat(weight("mm.gate.weight"), hidden))
	hidden = builder.MulMat(weight("mm.down.weight"), builder.Multiply(gate, up))
	boi := builder.Reshape(weight("v.boi"), uint64(r.spec.OutputHidden), tensor.SingletonExtent)
	eoi := builder.Reshape(weight("v.eoi"), uint64(r.spec.OutputHidden), tensor.SingletonExtent)
	output := builder.Concat(builder.Concat(boi, hidden, tensor.SingletonExtent), eoi, tensor.SingletonExtent)
	results, err := graph.execute(output)
	if err != nil {
		return reference.Value{}, fmt.Errorf("projector: execute CogVLM graph: %w", err)
	}
	return results[output], nil
}
