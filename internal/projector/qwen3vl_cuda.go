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

func (r *Qwen3VLRunner) encodeGraph(ctx context.Context, input Qwen3VLImage) (Qwen3VLOutput, error) {
	rows := input.GridT * input.GridH * input.GridW
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	temporalWidth := 3 * patchArea
	if len(input.PixelValues) != rows*temporalWidth*2 {
		return Qwen3VLOutput{}, errors.New("projector: Qwen3-VL input shape is inconsistent")
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
	hidden := builder.Add(builder.Add(builder.MulMat(patch0, input0), builder.MulMat(patch1, input1)), weight("v.patch_embd.bias"))
	rowOrder, columnOrder := mergedGrid(input.GridH, input.GridW, r.spec.MergeSize)
	hostFeeds := graph.hostFeeds
	hostFeeds[input0] = reference.Value{Shape: input0.Shape, Data: pixels0}
	hostFeeds[input1] = reference.Value{Shape: input1.Shape, Data: pixels1}
	hidden = r.qwen3VLPositionGraph(builder, hidden, weight("v.position_embd.weight"), input, rowOrder, columnOrder, hostFeeds)
	positionsY := make([]uint32, rows)
	positionsX := make([]uint32, rows)
	spatial := input.GridH * input.GridW
	for row := 0; row < rows; row++ {
		positionsY[row] = uint32(rowOrder[row%spatial])
		positionsX[row] = uint32(columnOrder[row%spatial])
	}
	var deepstack []*tensor.Tensor
	mergeFactor := r.spec.MergeSize * r.spec.MergeSize
	mergedRows := rows / mergeFactor
	mergedWidth := r.spec.Hidden * mergeFactor
	for layer := 0; layer < r.spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.AffineLayerNorm(hidden, weight(prefix+"ln1.weight"), weight(prefix+"ln1.bias"), r.spec.LayerNormEpsilon)
		qkv := builder.Add(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), weight(prefix+"attn_qkv.bias"))
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, 0, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(2*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
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
		up := builder.Add(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), weight(prefix+"ffn_up.bias"))
		up = qwen3VLGELUTanh(builder, up, hostFeeds)
		down := builder.Add(builder.MulMat(weight(prefix+"ffn_down.weight"), up), weight(prefix+"ffn_down.bias"))
		hidden = builder.Add(hidden, down)
		if len(r.spec.DeepstackLayers) > layer && r.spec.DeepstackLayers[layer] {
			prefix = fmt.Sprintf("v.deepstack.%d.", layer)
			merged := builder.Reshape(hidden, uint64(mergedWidth), uint64(mergedRows))
			norm = builder.AffineLayerNorm(merged, weight(prefix+"norm.weight"), weight(prefix+"norm.bias"), r.spec.LayerNormEpsilon)
			fc1 := builder.Add(builder.MulMat(weight(prefix+"fc1.weight"), norm), weight(prefix+"fc1.bias"))
			fc1 = qwen3VLGELUTanh(builder, fc1, hostFeeds)
			deepstack = append(deepstack, builder.Add(builder.MulMat(weight(prefix+"fc2.weight"), fc1), weight(prefix+"fc2.bias")))
		}
	}
	normalized := builder.AffineLayerNorm(hidden, weight("v.post_ln.weight"), weight("v.post_ln.bias"), r.spec.LayerNormEpsilon)
	merged := builder.Reshape(normalized, uint64(r.spec.Hidden*4), uint64(mergedRows))
	fc1 := builder.Add(builder.MulMat(weight("mm.0.weight"), merged), weight("mm.0.bias"))
	fc1 = qwen3VLGELUTanh(builder, fc1, hostFeeds)
	output := builder.Add(builder.MulMat(weight("mm.2.weight"), fc1), weight("mm.2.bias"))
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

func (r *Qwen3VLRunner) qwen3VLPositionGraph(
	builder *tensor.Builder,
	hidden, table *tensor.Tensor,
	input Qwen3VLImage,
	rowOrder, columnOrder []int,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	rows := input.GridT * input.GridH * input.GridW
	spatial := input.GridH * input.GridW
	side := r.spec.ImageSize / r.spec.PatchSize
	indexes := [4][]uint32{}
	weights := [4][]float32{}
	for corner := range indexes {
		indexes[corner] = make([]uint32, rows)
		weights[corner] = make([]float32, rows)
	}
	coordinate := func(index, extent int) float64 {
		if extent == 1 {
			return 0
		}
		return float64(side-1) * float64(index) / float64(extent-1)
	}
	for row := 0; row < rows; row++ {
		token := row % spatial
		y := coordinate(rowOrder[token], input.GridH)
		x := coordinate(columnOrder[token], input.GridW)
		y0, x0 := int(math.Floor(y)), int(math.Floor(x))
		y1, x1 := min(y0+1, side-1), min(x0+1, side-1)
		wy, wx := y-float64(y0), x-float64(x0)
		cornerIndexes := [4]int{y0*side + x0, y0*side + x1, y1*side + x0, y1*side + x1}
		cornerWeights := [4]float64{(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx}
		for corner := range indexes {
			indexes[corner][row] = uint32(cornerIndexes[corner])
			weights[corner][row] = float32(cornerWeights[corner])
		}
	}
	var position *tensor.Tensor
	for corner := range indexes {
		weight := builder.Input(fmt.Sprintf("position_weight.%d", corner), dtype.F32, tensor.MustShape(1, uint64(rows)))
		hostFeeds[weight] = reference.Value{Shape: weight.Shape, Data: weights[corner]}
		part := builder.Multiply(builder.GetRows(table, indexes[corner]), weight)
		if position == nil {
			position = part
		} else {
			position = builder.Add(position, part)
		}
	}
	return builder.Add(hidden, position)
}

func qwen3VLVisionRoPE(
	builder *tensor.Builder,
	input *tensor.Tensor,
	positionsY, positionsX []uint32,
) *tensor.Tensor {
	headWidth := input.Shape.Dims[0]
	quarter := headWidth / 4
	heads := input.Shape.Dims[1]
	rows := input.Shape.Dims[2]
	y := builder.Reshape(builder.GroupSlice(input, 0, quarter, 2, headWidth/2), headWidth/2, heads, rows)
	x := builder.Reshape(builder.GroupSlice(input, quarter, quarter, 2, headWidth/2), headWidth/2, heads, rows)
	y = builder.RoPENeoX(y, positionsY, uint32(headWidth/2), 10000)
	x = builder.RoPENeoX(x, positionsX, uint32(headWidth/2), 10000)
	split := func(value *tensor.Tensor, offset uint64) *tensor.Tensor {
		return builder.Reshape(builder.GroupSlice(value, offset, quarter, 1, quarter), quarter, heads, rows)
	}
	first := builder.Concat(split(y, 0), split(x, 0), 0)
	second := builder.Concat(split(y, quarter), split(x, quarter), 0)
	return builder.Concat(first, second, 0)
}

func qwen3VLGELUTanh(
	builder *tensor.Builder,
	input *tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	ones := builder.Input(fmt.Sprintf("gelu_ones.%d", input.ID), dtype.F32, tensor.MustShape(1))
	hostFeeds[ones] = reference.Value{Shape: ones.Shape, Data: []float32{1}}
	cube := builder.Multiply(builder.Multiply(input, input), input)
	inner := builder.Scale(builder.Add(input, builder.Scale(cube, 0.044715)), float32(math.Sqrt(2/math.Pi)))
	return builder.Scale(builder.Multiply(input, builder.Add(ones, builder.Tanh(inner))), 0.5)
}
