package projector

import (
	"context"
	"fmt"

	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *Granite4VisionRunner) encodeTileCUDA(ctx context.Context, tile Granite4VisionTile) ([]reference.Value, error) {
	side := r.spec.ImageSize / r.spec.PatchSize
	rows, patchWidth, err := validateSpatialPatchStorage(
		len(tile.PixelValues), side, side, r.spec.PatchSize, media.RGBChannels,
	)
	if err != nil {
		return nil, fmt.Errorf("projector: Granite 4 Vision CUDA tile: %w", err)
	}
	builder := tensor.NewBuilder()
	pixels := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[pixels] = pixelsValue(pixels, tile.PixelValues)
	weight := graph.weight
	patch := builder.Reshape(weight(visionPatchWeightTensor), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch, pixels), weight(visionPatchBiasTensor))
	hidden = builder.Add(hidden, weight(visionPositionWeightTensor))
	layerOutputs := make([]*tensor.Tensor, r.spec.Layers)
	headWidth := uint64(r.spec.Hidden / r.spec.Heads)
	for layer := range r.spec.Layers {
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
		up := builder.GELUTanhExact(graph.linear(norm, prefix+".ffn_up"))
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
		enc := windowSpatialRowsGraph(builder, x, side, windowSide)
		enc = builder.Reshape(enc, uint64(r.spec.Hidden), uint64(encLength), uint64(windows))
		enc = builder.Add(enc, weight(prefix+".img_pos"))
		enc = builder.Reshape(enc, uint64(r.spec.Hidden), uint64(windows*encLength))
		down := downsampleSpatialRowsGraph(builder, x, side, newSide, r.spec.SpatialOffsets[block])
		query := windowSpatialRowsGraph(builder, down, newSide, querySide)
		query = builder.Reshape(query, uint64(r.spec.Hidden), uint64(queryLength), uint64(windows))
		query = builder.Add(query, weight(prefix+".query"))
		query = builder.Reshape(query, uint64(r.spec.Hidden), uint64(windows*queryLength))
		query = graph.affineNorm(query, prefix+".post_norm", r.spec.ProjectionNormEpsilon)
		self := batchedAttentionGraph(graph, query, query, windows, queryLength, queryLength, r.spec.Hidden, r.spec.Heads, prefix+".self_attn")
		self = graph.affineNorm(builder.Add(query, self), prefix+".self_attn_norm", r.spec.ProjectionNormEpsilon)
		cross := batchedAttentionGraph(graph, self, enc, windows, queryLength, encLength, r.spec.Hidden, r.spec.Heads, prefix+".cross_attn")
		cross = graph.affineNorm(builder.Add(self, cross), prefix+".cross_attn_norm", r.spec.ProjectionNormEpsilon)
		ffn := graph.linear(builder.GELUErf(graph.linear(cross, prefix+".ffn_up")), prefix+".ffn_down")
		ffn = graph.affineNorm(builder.Add(cross, ffn), prefix+".ffn_norm", r.spec.ProjectionNormEpsilon)
		ffn = unwindowSpatialRowsGraph(builder, ffn, newSide, querySide)
		output := graph.linear(ffn, prefix+".linear")
		if tile.AddNewline {
			output = builder.Concat(output, builder.Reshape(weight(visionImageNewlineTensor),
				uint64(r.spec.ProjectionDim), tensor.SingletonExtent), tensor.SingletonExtent)
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

func windowSpatialRowsGraph(builder *tensor.Builder, input *tensor.Tensor, side, windowSide int) *tensor.Tensor {
	indices := make([]uint32, tensor.FirstOffset, side*side)
	for windowY := range side / windowSide {
		for windowX := range side / windowSide {
			for y := range windowSide {
				for x := range windowSide {
					indices = append(indices, uint32((windowY*windowSide+y)*side+windowX*windowSide+x))
				}
			}
		}
	}
	return builder.GetRows(input, indices)
}

func unwindowSpatialRowsGraph(builder *tensor.Builder, input *tensor.Tensor, side, windowSide int) *tensor.Tensor {
	indices := make([]uint32, side*side)
	source := tensor.FirstOffset
	for windowY := range side / windowSide {
		for windowX := range side / windowSide {
			for y := range windowSide {
				for x := range windowSide {
					indices[(windowY*windowSide+y)*side+windowX*windowSide+x] = uint32(source)
					source++
				}
			}
		}
	}
	return builder.GetRows(input, indices)
}

func downsampleSpatialRowsGraph(builder *tensor.Builder, input *tensor.Tensor, side, newSide, spatialOffset int) *tensor.Tensor {
	if spatialOffset >= tensor.FirstOffset {
		offsetY := (spatialOffset >> tensor.SingletonExtent) & tensor.SingletonExtent
		offsetX := spatialOffset & tensor.SingletonExtent
		indices := make([]uint32, tensor.FirstOffset, newSide*newSide)
		for y := range newSide {
			for x := range newSide {
				indices = append(indices, uint32((y*tensor.PairedExtent+offsetY)*side+x*tensor.PairedExtent+offsetX))
			}
		}
		return builder.GetRows(input, indices)
	}
	kernel := side / newSide
	var output *tensor.Tensor
	for ky := range kernel {
		for kx := range kernel {
			indices := make([]uint32, tensor.FirstOffset, newSide*newSide)
			for y := range newSide {
				for x := range newSide {
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
	return builder.Scale(output, float32(tensor.SingletonExtent)/float32(kernel*kernel))
}

func batchedAttentionGraph(
	graph *projectorGraphRuntime,
	queryInput, keyValueInput *tensor.Tensor,
	windows, queryRows, keyRows, hidden, headCount int,
	prefix string,
) *tensor.Tensor {
	builder := graph.builder
	query := graph.linear(queryInput, prefix+"_q")
	key := graph.linear(keyValueInput, prefix+"_k")
	value := graph.linear(keyValueInput, prefix+"_v")
	queryOrder := transposeBatchRowsOrder(windows, queryRows)
	keyOrder := transposeBatchRowsOrder(windows, keyRows)
	query = builder.GetRows(query, queryOrder)
	key = builder.GetRows(key, keyOrder)
	value = builder.GetRows(value, keyOrder)
	heads := headCount
	headWidth := uint64(hidden / heads)
	query = builder.Reshape(query, headWidth, uint64(heads*windows), uint64(queryRows))
	key = builder.Reshape(key, headWidth, uint64(heads*windows), uint64(keyRows))
	value = builder.Reshape(value, headWidth, uint64(heads*windows), uint64(keyRows))
	output := builder.AttentionWithOptions(query, key, value,
		tensor.AttentionOptions{Scale: hostmath.InvSqrt32(headWidth), Causal: false})
	output = builder.Reshape(output, uint64(hidden), uint64(windows*queryRows))
	inverse := make([]uint32, tensor.FirstOffset, windows*queryRows)
	for window := range windows {
		for row := range queryRows {
			inverse = append(inverse, uint32(row*windows+window))
		}
	}
	return graph.linear(builder.GetRows(output, inverse), prefix+"_out")
}

func transposeBatchRowsOrder(batches, rows int) []uint32 {
	indices := make([]uint32, tensor.FirstOffset, batches*rows)
	for row := range rows {
		for batch := range batches {
			indices = append(indices, uint32(batch*rows+row))
		}
	}
	return indices
}
