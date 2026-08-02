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

type cogVLMVisionCUDA struct {
	worker   *device.Worker
	executor *executor.Executor
	weights  *model.DeviceF32Weights
}

func openCogVLMVisionCUDA(ctx context.Context, file *gguf.File, spec CogVLMVisionSpec, ordinal int) (*cogVLMVisionCUDA, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	state := &cogVLMVisionCUDA{worker: worker}
	fail := func(cause error) (*cogVLMVisionCUDA, error) {
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
		"v.patch_embd.weight", "v.class_embd", "v.position_embd.weight",
		"mm.model.fc.weight", "mm.post_fc_norm.weight", "mm.post_fc_norm.bias",
		"mm.up.weight", "mm.gate.weight", "mm.down.weight", "v.boi", "v.eoi",
	}
	if hasTensor(file, "v.patch_embd.bias") {
		names = append(names, "v.patch_embd.bias")
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for _, suffix := range []string{
			"attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "attn_out.bias",
			"ffn_up.weight", "ffn_down.weight", "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
		} {
			names = append(names, prefix+suffix)
		}
		if spec.GatedFFN[layer] {
			names = append(names, prefix+"ffn_gate.weight")
		}
		for _, suffix := range []string{"ffn_up.bias", "ffn_gate.bias", "ffn_down.bias"} {
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

func (c *cogVLMVisionCUDA) Close() error {
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

func (r *CogVLMVisionRunner) encodeCUDA(ctx context.Context, pixelsData []float32) (reference.Value, error) {
	grid := r.spec.ImageSize / r.spec.PatchSize
	patchRows, patchWidth := grid*grid, 3*r.spec.PatchSize*r.spec.PatchSize
	rows := patchRows + 1
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(patchRows)))
	hostFeeds := map[*tensor.Tensor]reference.Value{pixels: pixelsValue(pixels, pixelsData)}
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
	addBias := func(value *tensor.Tensor, name string) *tensor.Tensor {
		if !hasTensor(r.file, name) {
			return value
		}
		return builder.Add(value, weight(name))
	}
	patch := builder.Reshape(weight("v.patch_embd.weight"), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := addBias(builder.MulMat(patch, pixels), "v.patch_embd.bias")
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
		up := addBias(builder.MulMat(weight(prefix+"ffn_up.weight"), hidden), prefix+"ffn_up.bias")
		if r.spec.GatedFFN[layer] {
			gate := addBias(builder.MulMat(weight(prefix+"ffn_gate.weight"), hidden), prefix+"ffn_gate.bias")
			up = builder.Multiply(up, qwen3VLGELUTanh(builder, gate, hostFeeds))
		} else {
			up = qwen3VLGELUTanh(builder, up, hostFeeds)
		}
		ffn := addBias(builder.MulMat(weight(prefix+"ffn_down.weight"), up), prefix+"ffn_down.bias")
		ffn = builder.AffineLayerNorm(ffn, weight(prefix+"ln2.weight"), weight(prefix+"ln2.bias"), r.spec.LayerNormEpsilon)
		hidden = builder.Add(hidden, ffn)
	}
	hidden = builder.FlatSlice(hidden, 0, uint64(r.spec.Hidden), uint64(patchRows))
	hidden = builder.MulMat(weight("mm.model.fc.weight"), hidden)
	hidden = builder.AffineLayerNorm(hidden, weight("mm.post_fc_norm.weight"), weight("mm.post_fc_norm.bias"), 1e-5)
	hidden = qwen3VLGELUTanh(builder, hidden, hostFeeds)
	up := builder.MulMat(weight("mm.up.weight"), hidden)
	gate := builder.SiLU(builder.MulMat(weight("mm.gate.weight"), hidden))
	hidden = builder.MulMat(weight("mm.down.weight"), builder.Multiply(gate, up))
	boi := builder.Reshape(weight("v.boi"), uint64(r.spec.OutputHidden), 1)
	eoi := builder.Reshape(weight("v.eoi"), uint64(r.spec.OutputHidden), 1)
	output := builder.Concat(builder.Concat(boi, hidden, 1), eoi, 1)
	if weightErr != nil {
		return reference.Value{}, fmt.Errorf("projector: build CogVLM CUDA graph: %w", weightErr)
	}
	if err := builder.Err(); err != nil {
		return reference.Value{}, fmt.Errorf("projector: build CogVLM CUDA graph: %w", err)
	}
	results, err := r.cuda.executor.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{output}, hostFeeds, deviceFeeds)
	if err != nil {
		return reference.Value{}, fmt.Errorf("projector: execute CogVLM CUDA graph: %w", err)
	}
	return results[output], nil
}
