package projector

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/cuda/executor"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

type hunyuanVLCUDA struct {
	worker   *device.Worker
	executor *executor.Executor
	weights  *model.DeviceF32Weights
}

func openHunyuanVLCUDA(ctx context.Context, file *gguf.File, spec HunyuanVLSpec, ordinal int) (*hunyuanVLCUDA, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	state := &hunyuanVLCUDA{worker: worker}
	fail := func(cause error) (*hunyuanVLCUDA, error) {
		_ = state.Close()
		return nil, cause
	}
	state.executor, err = executor.NewWithWorker(worker)
	if err != nil {
		return fail(err)
	}
	state.weights, err = model.NewDeviceF32Weights(worker)
	if err != nil {
		return fail(err)
	}
	names := []string{
		"v.patch_embd.weight", "v.position_embd.weight", "mm.pre_norm.weight",
		"mm.0.bias", "mm.2.weight", "mm.2.bias", "v.image_newline",
		"mm.model.fc.weight", "mm.model.fc.bias", "mm.image_begin", "mm.image_end", "mm.post_norm.weight",
	}
	for _, name := range []string{"v.patch_embd.bias", "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias"} {
		if hasTensor(file, name) {
			names = append(names, name)
		}
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		if spec.FusedQKV[layer] {
			names = append(names, prefix+"attn_qkv.weight")
			if hasTensor(file, prefix+"attn_qkv.bias") {
				names = append(names, prefix+"attn_qkv.bias")
			}
		} else {
			for _, part := range []string{"q", "k", "v"} {
				names = append(names, prefix+"attn_"+part+".weight")
				if hasTensor(file, prefix+"attn_"+part+".bias") {
					names = append(names, prefix+"attn_"+part+".bias")
				}
			}
		}
		for _, suffix := range []string{"attn_out.weight", "ffn_up.weight", "ffn_down.weight", "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"} {
			names = append(names, prefix+suffix)
		}
		for _, suffix := range []string{"attn_out.bias", "ffn_up.bias", "ffn_down.bias"} {
			if hasTensor(file, prefix+suffix) {
				names = append(names, prefix+suffix)
			}
		}
	}
	infos := make([]gguf.TensorInfo, len(names))
	for index, name := range names {
		info, ok := file.Tensor(name)
		if !ok {
			return fail(fmt.Errorf("tensor %q is unavailable", name))
		}
		infos[index] = info
	}
	if err := state.weights.Load(ctx, file, infos); err != nil {
		return fail(err)
	}
	return state, nil
}

func (c *hunyuanVLCUDA) Close() error {
	if c == nil {
		return nil
	}
	var closeErrors []error
	if c.weights != nil {
		closeErrors = append(closeErrors, c.weights.Close())
		c.weights = nil
	}
	if c.executor != nil {
		closeErrors = append(closeErrors, c.executor.Close())
		c.executor = nil
	}
	if c.worker != nil {
		closeErrors = append(closeErrors, c.worker.Close())
		c.worker = nil
	}
	return errors.Join(closeErrors...)
}

func (r *HunyuanVLRunner) encodeCUDA(ctx context.Context, input HunyuanVLImage) (HunyuanVLOutput, error) {
	rows := input.GridH * input.GridW
	patchWidth := 3 * r.spec.PatchSize * r.spec.PatchSize
	if rows <= 0 || input.GridH%r.spec.MergeSize != 0 || input.GridW%r.spec.MergeSize != 0 || len(input.PixelValues) != rows*patchWidth {
		return HunyuanVLOutput{}, errors.New("projector: Hunyuan-VL CUDA input shape is inconsistent")
	}
	conv0, err := r.load(ctx, "mm.0.weight")
	if err != nil {
		return HunyuanVLOutput{}, err
	}
	mergeFactor := r.spec.MergeSize * r.spec.MergeSize
	convWidth := r.spec.Hidden * mergeFactor
	reordered := make([]float32, len(conv0.Data))
	for output := 0; output < r.spec.ConvIntermediate; output++ {
		for offset := 0; offset < mergeFactor; offset++ {
			for channel := 0; channel < r.spec.Hidden; channel++ {
				source := output*convWidth + channel*mergeFactor + offset
				destination := output*convWidth + offset*r.spec.Hidden + channel
				reordered[destination] = conv0.Data[source]
			}
		}
	}
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	conv0Input := builder.Input("mm.0.weight.reordered", dtype.F32, tensor.MustShape(uint64(convWidth), uint64(r.spec.ConvIntermediate)))
	hostFeeds := map[*tensor.Tensor]reference.Value{
		pixels: pixelsValue(pixels, input.PixelValues), conv0Input: pixelsValue(conv0Input, reordered),
	}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var weightErr error
	weight := func(name string) *tensor.Tensor {
		node, pointer, inputErr := r.cuda.weights.Input(builder, name)
		if inputErr != nil {
			weightErr = inputErr
			return nil
		}
		deviceFeeds[node] = pointer
		return node
	}
	addBias := func(value *tensor.Tensor, name string) *tensor.Tensor {
		if !hasTensor(r.file, name) {
			return value
		}
		return builder.Add(value, weight(name))
	}
	patch := builder.Reshape(weight("v.patch_embd.weight"), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := addBias(builder.MulMat(patch, pixels), "v.patch_embd.bias")
	hidden = r.hunyuanVLPositionGraph(builder, hidden, weight("v.position_embd.weight"), input.GridH, input.GridW, hostFeeds)
	if r.spec.PreLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.pre_ln.weight"), weight("v.pre_ln.bias"), r.spec.LayerNormEpsilon)
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
		attention := builder.Attention(q, k, v, float32(1/math.Sqrt(float64(headWidth))), false)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		projected := addBias(builder.MulMat(weight(prefix+"attn_out.weight"), attention), prefix+"attn_out.bias")
		hidden = builder.Add(hidden, projected)
		norm = builder.AffineLayerNorm(hidden, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		up := addBias(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), prefix+"ffn_up.bias")
		up = qwen3VLGELUTanh(builder, up, hostFeeds)
		down := addBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		hidden = builder.Add(hidden, down)
	}
	if r.spec.PostLayerNorm {
		hidden = builder.AffineLayerNorm(hidden, weight("v.post_ln.weight"), weight("v.post_ln.bias"), r.spec.LayerNormEpsilon)
	}
	hidden = builder.WeightedRMSNorm(hidden, weight("mm.pre_norm.weight"), r.spec.LayerNormEpsilon)
	mergedH, mergedW := input.GridH/r.spec.MergeSize, input.GridW/r.spec.MergeSize
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
	projected := builder.Add(builder.MulMat(conv0Input, merged), weight("mm.0.bias"))
	projected = qwen3VLGELUTanh(builder, projected, hostFeeds)
	conv2 := builder.Reshape(weight("mm.2.weight"), uint64(r.spec.ConvIntermediate), uint64(r.spec.ProjectorInput))
	projected = builder.Add(builder.MulMat(conv2, projected), weight("mm.2.bias"))
	newline := builder.Reshape(weight("v.image_newline"), uint64(r.spec.ProjectorInput), 1)
	var content *tensor.Tensor
	for y := 0; y < mergedH; y++ {
		row := builder.FlatSlice(projected, uint64(y*mergedW*r.spec.ProjectorInput), uint64(r.spec.ProjectorInput), uint64(mergedW))
		row = builder.Concat(row, newline, 1)
		if content == nil {
			content = row
		} else {
			content = builder.Concat(content, row, 1)
		}
	}
	content = builder.Add(builder.MulMat(weight("mm.model.fc.weight"), content), weight("mm.model.fc.bias"))
	begin := builder.Reshape(weight("mm.image_begin"), uint64(r.spec.OutputHidden), 1)
	end := builder.Reshape(weight("mm.image_end"), uint64(r.spec.OutputHidden), 1)
	output := builder.Concat(builder.Concat(begin, content, 1), end, 1)
	output = builder.WeightedRMSNorm(output, weight("mm.post_norm.weight"), r.spec.LayerNormEpsilon)
	if weightErr != nil {
		return HunyuanVLOutput{}, fmt.Errorf("projector: build Hunyuan-VL CUDA graph: %w", weightErr)
	}
	if err := builder.Err(); err != nil {
		return HunyuanVLOutput{}, fmt.Errorf("projector: build Hunyuan-VL CUDA graph: %w", err)
	}
	results, err := r.cuda.executor.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{output}, hostFeeds, deviceFeeds)
	if err != nil {
		return HunyuanVLOutput{}, fmt.Errorf("projector: execute Hunyuan-VL CUDA graph: %w", err)
	}
	return HunyuanVLOutput{Embeddings: results[output], GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}

func pixelsValue(node *tensor.Tensor, data []float32) reference.Value {
	return reference.Value{Shape: node.Shape, Data: data}
}

func (r *HunyuanVLRunner) hunyuanVLPositionGraph(
	builder *tensor.Builder,
	hidden, table *tensor.Tensor,
	gridH, gridW int,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	rows := gridH * gridW
	side := r.spec.ImageSize / r.spec.PatchSize
	sx := (float64(gridW) + 0.1) / float64(side)
	sy := (float64(gridH) + 0.1) / float64(side)
	indexes := [4][]uint32{}
	weights := [4][]float32{}
	for corner := range indexes {
		indexes[corner] = make([]uint32, rows)
		weights[corner] = make([]float32, rows)
	}
	for y := 0; y < gridH; y++ {
		fy := (float64(y)+0.5)/sy - 0.5
		y0 := max(0, min(side-1, int(math.Floor(fy))))
		y1 := max(0, min(side-1, int(math.Floor(fy))+1))
		wy := min(1.0, max(0.0, fy-float64(y0)))
		for x := 0; x < gridW; x++ {
			fx := (float64(x)+0.5)/sx - 0.5
			x0 := max(0, min(side-1, int(math.Floor(fx))))
			x1 := max(0, min(side-1, int(math.Floor(fx))+1))
			wx := min(1.0, max(0.0, fx-float64(x0)))
			row := y*gridW + x
			cornerIndexes := [4]int{y0*side + x0, y0*side + x1, y1*side + x0, y1*side + x1}
			cornerWeights := [4]float64{(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx}
			for corner := range indexes {
				indexes[corner][row] = uint32(cornerIndexes[corner])
				weights[corner][row] = float32(cornerWeights[corner])
			}
		}
	}
	var position *tensor.Tensor
	for corner := range indexes {
		factor := builder.Input(fmt.Sprintf("hunyuan_position_weight.%d", corner), dtype.F32, tensor.MustShape(1, uint64(rows)))
		hostFeeds[factor] = pixelsValue(factor, weights[corner])
		part := builder.Multiply(builder.GetRows(table, indexes[corner]), factor)
		if position == nil {
			position = part
		} else {
			position = builder.Add(position, part)
		}
	}
	return builder.Add(hidden, position)
}
