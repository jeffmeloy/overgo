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

func openQwen2VLCUDA(ctx context.Context, file *gguf.File, spec Qwen2VLSpec, ordinal int) (*qwen3VLCUDA, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	state := &qwen3VLCUDA{worker: worker}
	fail := func(cause error) (*qwen3VLCUDA, error) {
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
		"v.patch_embd.weight", "v.patch_embd.weight.1",
		"mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias",
	}
	if spec.PreLayerNorm {
		names = append(names, "v.pre_ln.weight", "v.pre_ln.bias")
	}
	if spec.PostLayerNorm {
		names = append(names, "v.post_ln.weight", "v.post_ln.bias")
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for _, suffix := range []string{
			"attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias",
			"attn_v.weight", "attn_v.bias", "attn_out.weight", "attn_out.bias",
			"ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias",
			"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
		} {
			names = append(names, prefix+suffix)
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

func (r *Qwen2VLRunner) encodeCUDA(ctx context.Context, input Qwen2VLImage) (Qwen2VLOutput, error) {
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
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var weightErr error
	weight := func(name string) *tensor.Tensor {
		node, pointer, err := r.cuda.weights.Input(builder, name)
		if err != nil {
			weightErr = err
			return nil
		}
		deviceFeeds[node] = pointer
		return node
	}
	patch0 := builder.Reshape(weight("v.patch_embd.weight"), uint64(temporalWidth), uint64(r.spec.Hidden))
	patch1 := builder.Reshape(weight("v.patch_embd.weight.1"), uint64(temporalWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch0, input0), builder.MulMat(patch1, input1))
	hostFeeds := map[*tensor.Tensor]reference.Value{
		input0: {Shape: input0.Shape, Data: pixels0}, input1: {Shape: input1.Shape, Data: pixels1},
	}
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
	if weightErr != nil {
		return Qwen2VLOutput{}, fmt.Errorf("projector: build Qwen2-VL CUDA graph: %w", weightErr)
	}
	if err := builder.Err(); err != nil {
		return Qwen2VLOutput{}, fmt.Errorf("projector: build Qwen2-VL CUDA graph: %w", err)
	}
	results, err := r.cuda.executor.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{output}, hostFeeds, deviceFeeds)
	if err != nil {
		return Qwen2VLOutput{}, fmt.Errorf("projector: execute Qwen2-VL CUDA graph: %w", err)
	}
	return Qwen2VLOutput{Embeddings: results[output], GridT: input.GridT, GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}
