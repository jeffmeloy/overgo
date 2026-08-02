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

type mimoVLCUDA struct {
	worker   *device.Worker
	executor *executor.Executor
	weights  *model.DeviceF32Weights
}

func openMiMoVLCUDA(ctx context.Context, file *gguf.File, spec MiMoVLSpec, ordinal int) (*mimoVLCUDA, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	state := &mimoVLCUDA{worker: worker}
	fail := func(cause error) (*mimoVLCUDA, error) {
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
	names := []string{"v.patch_embd.weight", "v.patch_embd.weight.1", "v.post_ln.weight", "mm.0.weight", "mm.2.weight"}
	for _, name := range []string{"v.post_ln.bias", "mm.0.bias", "mm.2.bias"} {
		if hasTensor(file, name) {
			names = append(names, name)
		}
	}
	for layer, mode := range spec.WindowModes {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for _, suffix := range []string{
			"attn_qkv.weight", "attn_qkv.bias", "attn_out.weight",
			"ffn_up.weight", "ffn_up.bias", "ffn_gate.weight", "ffn_gate.bias",
			"ffn_down.weight", "ffn_down.bias", "ln1.weight", "ln2.weight",
		} {
			names = append(names, prefix+suffix)
		}
		for _, suffix := range []string{"attn_out.bias", "ln1.bias", "ln2.bias"} {
			if hasTensor(file, prefix+suffix) {
				names = append(names, prefix+suffix)
			}
		}
		if mode != -1 {
			names = append(names, prefix+"attn_sinks")
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

func (c *mimoVLCUDA) Close() error {
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

func (r *MiMoVLRunner) encodeCUDA(ctx context.Context, input MiMoVLInput) (MiMoVLOutput, error) {
	rows := input.GridH * input.GridW
	if rows <= 0 || input.GridH%r.spec.MergeSize != 0 || input.GridW%r.spec.MergeSize != 0 {
		return MiMoVLOutput{}, errors.New("projector: MiMo-VL input geometry is inconsistent")
	}
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	temporalWidth := 3 * patchArea
	if len(input.PixelValues) != rows*temporalWidth*2 {
		return MiMoVLOutput{}, errors.New("projector: MiMo-VL pixel tensor is inconsistent")
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
	hostFeeds := map[*tensor.Tensor]reference.Value{
		input0: {Shape: input0.Shape, Data: pixels0}, input1: {Shape: input1.Shape, Data: pixels1},
	}
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
	addOptional := func(input *tensor.Tensor, name string) *tensor.Tensor {
		if hasTensor(r.file, name) {
			return builder.Add(input, weight(name))
		}
		return input
	}
	rmsNorm := func(input *tensor.Tensor, prefix string) *tensor.Tensor {
		output := builder.WeightedRMSNorm(input, weight(prefix+".weight"), r.spec.LayerNormEpsilon)
		return addOptional(output, prefix+".bias")
	}
	patch0 := builder.Reshape(weight("v.patch_embd.weight"), uint64(temporalWidth), uint64(r.spec.Hidden))
	patch1 := builder.Reshape(weight("v.patch_embd.weight.1"), uint64(temporalWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch0, input0), builder.MulMat(patch1, input1))
	rowPositions, columnPositions := mergedGrid(input.GridH, input.GridW, r.spec.MergeSize)
	positionsH, positionsW := intsToUint32(rowPositions), intsToUint32(columnPositions)
	columnOrder := mimoVLColumnOrder(input.GridH/r.spec.MergeSize, input.GridW/r.spec.MergeSize, r.spec.MergeSize)
	inverseColumnOrder := inversePermutation(columnOrder)
	previousMode := -1
	qWidth := r.spec.Heads * r.spec.HeadDim
	kvWidth := r.spec.KVHeads * r.spec.HeadDim
	for layer, mode := range r.spec.WindowModes {
		if mode == 1 && previousMode != 1 {
			hidden = builder.GetRows(hidden, intsToUint32(columnOrder))
			positionsH = intsToUint32(reorderInts(rowPositions, columnOrder))
			positionsW = intsToUint32(reorderInts(columnPositions, columnOrder))
		} else if mode != 1 && previousMode == 1 {
			hidden = builder.GetRows(hidden, intsToUint32(inverseColumnOrder))
			positionsH, positionsW = intsToUint32(rowPositions), intsToUint32(columnPositions)
		}
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := rmsNorm(hidden, prefix+"ln1")
		qkv := builder.Add(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), weight(prefix+"attn_qkv.bias"))
		headDim := uint64(r.spec.HeadDim)
		q := builder.GroupSlice(qkv, 0, headDim, uint64(r.spec.Heads), headDim)
		k := builder.GroupSlice(qkv, uint64(qWidth), headDim, uint64(r.spec.KVHeads), headDim)
		v := builder.GroupSlice(qkv, uint64(qWidth+kvWidth), headDim, uint64(r.spec.KVHeads), headDim)
		q = qwen3VLVisionRoPE(builder, q, positionsH, positionsW)
		k = qwen3VLVisionRoPE(builder, k, positionsH, positionsW)
		scale := float32(1 / math.Sqrt(float64(r.spec.HeadDim)))
		var attention *tensor.Tensor
		if mode == -1 {
			attention = builder.Attention(q, k, v, scale, false)
		} else {
			attention = builder.AttentionSymmetricWindowWithSinks(
				q, k, v, weight(prefix+"attn_sinks"), scale, uint32(2*r.spec.WindowSize),
			)
		}
		attention = builder.Reshape(attention, uint64(qWidth), uint64(rows))
		projected := builder.MulMat(weight(prefix+"attn_out.weight"), attention)
		hidden = builder.Add(hidden, addOptional(projected, prefix+"attn_out.bias"))
		norm = rmsNorm(hidden, prefix+"ln2")
		up := builder.Add(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), weight(prefix+"ffn_up.bias"))
		gate := builder.Add(builder.MulMat(weight(prefix+"ffn_gate.weight"), norm), weight(prefix+"ffn_gate.bias"))
		activated := builder.Multiply(builder.SiLU(gate), up)
		down := builder.Add(builder.MulMat(weight(prefix+"ffn_down.weight"), activated), weight(prefix+"ffn_down.bias"))
		hidden = builder.Add(hidden, down)
		previousMode = mode
	}
	if previousMode == 1 {
		hidden = builder.GetRows(hidden, intsToUint32(inverseColumnOrder))
	}
	normalized := builder.Multiply(builder.LayerNorm(hidden, 1e-6), weight("v.post_ln.weight"))
	normalized = addOptional(normalized, "v.post_ln.bias")
	mergedRows := rows / (r.spec.MergeSize * r.spec.MergeSize)
	merged := builder.Reshape(normalized, uint64(r.spec.Hidden*4), uint64(mergedRows))
	fc1 := builder.MulMat(weight("mm.0.weight"), merged)
	fc1 = qwen3VLGELUTanh(builder, addOptional(fc1, "mm.0.bias"), hostFeeds)
	output := builder.MulMat(weight("mm.2.weight"), fc1)
	output = addOptional(output, "mm.2.bias")
	if weightErr != nil {
		return MiMoVLOutput{}, fmt.Errorf("projector: build MiMo-VL CUDA graph: %w", weightErr)
	}
	if err := builder.Err(); err != nil {
		return MiMoVLOutput{}, fmt.Errorf("projector: build MiMo-VL CUDA graph: %w", err)
	}
	results, err := r.cuda.executor.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{output}, hostFeeds, deviceFeeds)
	if err != nil {
		return MiMoVLOutput{}, fmt.Errorf("projector: execute MiMo-VL CUDA graph: %w", err)
	}
	return MiMoVLOutput{
		Embeddings: results[output], GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize,
	}, nil
}

func intsToUint32(values []int) []uint32 {
	output := make([]uint32, len(values))
	for index, value := range values {
		output[index] = uint32(value)
	}
	return output
}
