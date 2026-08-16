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

func (r *Granite4VisionRunner) encodeTileCUDA(ctx context.Context, tile Granite4VisionTile) ([]reference.Value, error) {
	side := r.spec.ImageSize / r.spec.PatchSize
	rows, patchWidth := side*side, 3*r.spec.PatchSize*r.spec.PatchSize
	if len(tile.PixelValues) != rows*patchWidth {
		return nil, errors.New("projector: Granite 4 Vision CUDA tile shape is inconsistent")
	}
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	hostFeeds := graph.hostFeeds
	hostFeeds[pixels] = pixelsValue(pixels, tile.PixelValues)
	weight := graph.weight
	patch := builder.Reshape(weight(visionPatchWeightTensor), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch, pixels), weight(visionPatchBiasTensor))
	hidden = builder.Add(hidden, weight(visionPositionWeightTensor))
	layerOutputs := make([]*tensor.Tensor, r.spec.Layers)
	headWidth := uint64(r.spec.Hidden / r.spec.Heads)
	for layer := 0; layer < r.spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d", layer)
		norm := graph.affineNorm(hidden, prefix+".ln1", r.spec.LayerNormEpsilon)
		q := graph.linear(norm, prefix+".attn_q")
		k := graph.linear(norm, prefix+".attn_k")
		v := graph.linear(norm, prefix+".attn_v")
		q = builder.Reshape(q, headWidth, uint64(r.spec.Heads), uint64(rows))
		k = builder.Reshape(k, headWidth, uint64(r.spec.Heads), uint64(rows))
		v = builder.Reshape(v, headWidth, uint64(r.spec.Heads), uint64(rows))
		attention := r.attention.graph(builder, q, k, v)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		hidden = builder.Add(hidden, graph.linear(attention, prefix+".attn_out"))
		norm = graph.affineNorm(hidden, prefix+".ln2", r.spec.LayerNormEpsilon)
		up := qwen3VLGELUTanh(builder, graph.linear(norm, prefix+".ffn_up"), hostFeeds)
		hidden = builder.Add(hidden, graph.linear(up, prefix+".ffn_down"))
		layerOutputs[layer] = hidden
	}
	outputs := make([]*tensor.Tensor, len(r.spec.FeatureLayers))
	for block, layer := range r.spec.FeatureLayers {
		prefix := fmt.Sprintf("v.proj_blk.%d", block)
		x := graph.affineNorm(layerOutputs[layer], prefix+".norm", r.spec.LayerNormEpsilon)
		windowSide, querySide := r.spec.WindowSide, r.spec.QuerySide
		windowsPerSide := side / windowSide
		windows := windowsPerSide * windowsPerSide
		encLength, queryLength := windowSide*windowSide, querySide*querySide
		newSide := windowsPerSide * querySide
		enc := granite4WindowGraph(builder, x, side, windowSide)
		enc = builder.Reshape(enc, uint64(r.spec.Hidden), uint64(encLength), uint64(windows))
		enc = builder.Add(enc, weight(prefix+".img_pos"))
		enc = builder.Reshape(enc, uint64(r.spec.Hidden), uint64(windows*encLength))
		down := granite4DownsampleGraph(builder, x, side, newSide, r.spec.SpatialOffsets[block])
		query := granite4WindowGraph(builder, down, newSide, querySide)
		query = builder.Reshape(query, uint64(r.spec.Hidden), uint64(queryLength), uint64(windows))
		query = builder.Add(query, weight(prefix+".query"))
		query = builder.Reshape(query, uint64(r.spec.Hidden), uint64(windows*queryLength))
		query = graph.affineNorm(query, prefix+".post_norm", granite4QFormerNormEpsilon)
		self := granite4AttentionGraph(graph, query, query, windows, queryLength, queryLength, r.spec.Hidden, prefix+".self_attn")
		self = graph.affineNorm(builder.Add(query, self), prefix+".self_attn_norm", granite4QFormerNormEpsilon)
		cross := granite4AttentionGraph(graph, self, enc, windows, queryLength, encLength, r.spec.Hidden, prefix+".cross_attn")
		cross = graph.affineNorm(builder.Add(self, cross), prefix+".cross_attn_norm", granite4QFormerNormEpsilon)
		ffn := graph.linear(builder.GELUErf(graph.linear(cross, prefix+".ffn_up")), prefix+".ffn_down")
		ffn = graph.affineNorm(builder.Add(cross, ffn), prefix+".ffn_norm", granite4QFormerNormEpsilon)
		ffn = granite4UnwindowGraph(builder, ffn, newSide, querySide)
		output := graph.linear(ffn, prefix+".linear")
		if tile.AddNewline {
			output = builder.Concat(output, builder.Reshape(weight("v.image_newline"), uint64(r.spec.ProjectionDim), 1), 1)
		}
		outputs[block] = output
	}
	results, err := graph.execute(outputs...)
	if err != nil {
		return nil, fmt.Errorf("projector: execute Granite 4 Vision CUDA graph: %w", err)
	}
	values := make([]reference.Value, len(outputs))
	for index, output := range outputs {
		values[index] = results[output]
	}
	return values, nil
}

func granite4WindowGraph(builder *tensor.Builder, input *tensor.Tensor, side, windowSide int) *tensor.Tensor {
	indices := make([]uint32, 0, side*side)
	for windowY := 0; windowY < side/windowSide; windowY++ {
		for windowX := 0; windowX < side/windowSide; windowX++ {
			for y := 0; y < windowSide; y++ {
				for x := 0; x < windowSide; x++ {
					indices = append(indices, uint32((windowY*windowSide+y)*side+windowX*windowSide+x))
				}
			}
		}
	}
	return builder.GetRows(input, indices)
}

func granite4UnwindowGraph(builder *tensor.Builder, input *tensor.Tensor, side, windowSide int) *tensor.Tensor {
	indices := make([]uint32, side*side)
	source := 0
	for windowY := 0; windowY < side/windowSide; windowY++ {
		for windowX := 0; windowX < side/windowSide; windowX++ {
			for y := 0; y < windowSide; y++ {
				for x := 0; x < windowSide; x++ {
					indices[(windowY*windowSide+y)*side+windowX*windowSide+x] = uint32(source)
					source++
				}
			}
		}
	}
	return builder.GetRows(input, indices)
}

func granite4DownsampleGraph(builder *tensor.Builder, input *tensor.Tensor, side, newSide, spatialOffset int) *tensor.Tensor {
	if spatialOffset >= 0 {
		offsetY, offsetX := (spatialOffset>>1)&1, spatialOffset&1
		indices := make([]uint32, 0, newSide*newSide)
		for y := 0; y < newSide; y++ {
			for x := 0; x < newSide; x++ {
				indices = append(indices, uint32((y*2+offsetY)*side+x*2+offsetX))
			}
		}
		return builder.GetRows(input, indices)
	}
	kernel := side / newSide
	var output *tensor.Tensor
	for ky := 0; ky < kernel; ky++ {
		for kx := 0; kx < kernel; kx++ {
			indices := make([]uint32, 0, newSide*newSide)
			for y := 0; y < newSide; y++ {
				for x := 0; x < newSide; x++ {
					indices = append(indices, uint32((y*kernel+ky)*side+x*kernel+kx))
				}
			}
			part := builder.GetRows(input, indices)
			if output == nil {
				output = part
			} else {
				output = builder.Add(output, part)
			}
		}
	}
	return builder.Scale(output, 1/float32(kernel*kernel))
}

func granite4AttentionGraph(
	graph *projectorGraphRuntime,
	queryInput, keyValueInput *tensor.Tensor,
	windows, queryRows, keyRows, hidden int,
	prefix string,
) *tensor.Tensor {
	builder := graph.builder
	query := graph.linear(queryInput, prefix+"_q")
	key := graph.linear(keyValueInput, prefix+"_k")
	value := graph.linear(keyValueInput, prefix+"_v")
	queryOrder := granite4AttentionOrder(windows, queryRows)
	keyOrder := granite4AttentionOrder(windows, keyRows)
	query = builder.GetRows(query, queryOrder)
	key = builder.GetRows(key, keyOrder)
	value = builder.GetRows(value, keyOrder)
	heads := hidden / granite4VisionAttentionHeadWidth
	headWidth := uint64(granite4VisionAttentionHeadWidth)
	query = builder.Reshape(query, headWidth, uint64(heads*windows), uint64(queryRows))
	key = builder.Reshape(key, headWidth, uint64(heads*windows), uint64(keyRows))
	value = builder.Reshape(value, headWidth, uint64(heads*windows), uint64(keyRows))
	output := builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{Scale: float32(1 / math.Sqrt(float64(headWidth))), Causal: false})
	output = builder.Reshape(output, uint64(hidden), uint64(windows*queryRows))
	inverse := make([]uint32, 0, windows*queryRows)
	for window := 0; window < windows; window++ {
		for row := 0; row < queryRows; row++ {
			inverse = append(inverse, uint32(row*windows+window))
		}
	}
	return graph.linear(builder.GetRows(output, inverse), prefix+"_out")
}

func granite4AttentionOrder(windows, rows int) []uint32 {
	indices := make([]uint32, 0, windows*rows)
	for row := 0; row < rows; row++ {
		for window := 0; window < windows; window++ {
			indices = append(indices, uint32(window*rows+row))
		}
	}
	return indices
}
