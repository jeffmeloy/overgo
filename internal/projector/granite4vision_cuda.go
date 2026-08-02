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

type granite4VisionCUDA struct {
	worker   *device.Worker
	executor *executor.Executor
	weights  *model.DeviceF32Weights
}

func openGranite4VisionCUDA(ctx context.Context, file *gguf.File, spec Granite4VisionSpec, ordinal int) (*granite4VisionCUDA, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	state := &granite4VisionCUDA{worker: worker}
	fail := func(cause error) (*granite4VisionCUDA, error) {
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
	names := []string{"v.patch_embd.weight", "v.patch_embd.bias", "v.position_embd.weight", "v.image_newline"}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for _, suffix := range []string{
			"attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias", "attn_v.weight", "attn_v.bias",
			"attn_out.weight", "attn_out.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias",
			"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
		} {
			names = append(names, prefix+suffix)
		}
	}
	for block := range spec.FeatureLayers {
		prefix := fmt.Sprintf("v.proj_blk.%d.", block)
		names = append(names, prefix+"img_pos", prefix+"query", prefix+"linear.weight", prefix+"linear.bias")
		for _, name := range []string{"norm", "post_norm", "self_attn_norm", "cross_attn_norm", "ffn_norm"} {
			names = append(names, prefix+name+".weight", prefix+name+".bias")
		}
		for _, name := range []string{
			"self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out",
			"cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out",
		} {
			names = append(names, prefix+name+".weight", prefix+name+".bias")
		}
		names = append(names,
			prefix+"ffn_up.weight", prefix+"ffn_up.bias",
			prefix+"ffn_down.weight", prefix+"ffn_down.bias",
		)
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

func (c *granite4VisionCUDA) Close() error {
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

func (r *Granite4VisionRunner) encodeTileCUDA(ctx context.Context, tile Granite4VisionTile) ([]reference.Value, error) {
	side := r.spec.ImageSize / r.spec.PatchSize
	rows, patchWidth := side*side, 3*r.spec.PatchSize*r.spec.PatchSize
	if len(tile.PixelValues) != rows*patchWidth {
		return nil, errors.New("projector: Granite 4 Vision CUDA tile shape is inconsistent")
	}
	builder := tensor.NewBuilder()
	pixels := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	hostFeeds := map[*tensor.Tensor]reference.Value{pixels: pixelsValue(pixels, tile.PixelValues)}
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
	linear := func(input *tensor.Tensor, prefix string) *tensor.Tensor {
		return builder.Add(builder.MulMat(weight(prefix+".weight"), input), weight(prefix+".bias"))
	}
	affineNorm := func(input *tensor.Tensor, prefix string, epsilon float32) *tensor.Tensor {
		return builder.AffineLayerNorm(input, weight(prefix+".weight"), weight(prefix+".bias"), epsilon)
	}
	patch := builder.Reshape(weight("v.patch_embd.weight"), uint64(patchWidth), uint64(r.spec.Hidden))
	hidden := builder.Add(builder.MulMat(patch, pixels), weight("v.patch_embd.bias"))
	hidden = builder.Add(hidden, weight("v.position_embd.weight"))
	layerOutputs := make([]*tensor.Tensor, r.spec.Layers)
	headWidth := uint64(r.spec.Hidden / r.spec.Heads)
	for layer := 0; layer < r.spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d", layer)
		norm := affineNorm(hidden, prefix+".ln1", r.spec.LayerNormEpsilon)
		q := linear(norm, prefix+".attn_q")
		k := linear(norm, prefix+".attn_k")
		v := linear(norm, prefix+".attn_v")
		q = builder.Reshape(q, headWidth, uint64(r.spec.Heads), uint64(rows))
		k = builder.Reshape(k, headWidth, uint64(r.spec.Heads), uint64(rows))
		v = builder.Reshape(v, headWidth, uint64(r.spec.Heads), uint64(rows))
		attention := builder.Attention(q, k, v, float32(1/math.Sqrt(float64(headWidth))), false)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), uint64(rows))
		hidden = builder.Add(hidden, linear(attention, prefix+".attn_out"))
		norm = affineNorm(hidden, prefix+".ln2", r.spec.LayerNormEpsilon)
		up := qwen3VLGELUTanh(builder, linear(norm, prefix+".ffn_up"), hostFeeds)
		hidden = builder.Add(hidden, linear(up, prefix+".ffn_down"))
		layerOutputs[layer] = hidden
	}
	outputs := make([]*tensor.Tensor, len(r.spec.FeatureLayers))
	for block, layer := range r.spec.FeatureLayers {
		prefix := fmt.Sprintf("v.proj_blk.%d", block)
		x := affineNorm(layerOutputs[layer], prefix+".norm", r.spec.LayerNormEpsilon)
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
		query = affineNorm(query, prefix+".post_norm", 1e-12)
		self := granite4AttentionGraph(builder, query, query, windows, queryLength, queryLength, r.spec.Hidden, prefix+".self_attn", linear)
		self = affineNorm(builder.Add(query, self), prefix+".self_attn_norm", 1e-12)
		cross := granite4AttentionGraph(builder, self, enc, windows, queryLength, encLength, r.spec.Hidden, prefix+".cross_attn", linear)
		cross = affineNorm(builder.Add(self, cross), prefix+".cross_attn_norm", 1e-12)
		ffn := linear(builder.GELUErf(linear(cross, prefix+".ffn_up")), prefix+".ffn_down")
		ffn = affineNorm(builder.Add(cross, ffn), prefix+".ffn_norm", 1e-12)
		ffn = granite4UnwindowGraph(builder, ffn, newSide, querySide)
		output := linear(ffn, prefix+".linear")
		if tile.AddNewline {
			output = builder.Concat(output, builder.Reshape(weight("v.image_newline"), uint64(r.spec.ProjectionDim), 1), 1)
		}
		outputs[block] = output
	}
	if weightErr != nil {
		return nil, fmt.Errorf("projector: build Granite 4 Vision CUDA graph: %w", weightErr)
	}
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("projector: build Granite 4 Vision CUDA graph: %w", err)
	}
	results, err := r.cuda.executor.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
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
	builder *tensor.Builder,
	queryInput, keyValueInput *tensor.Tensor,
	windows, queryRows, keyRows, hidden int,
	prefix string,
	linear func(*tensor.Tensor, string) *tensor.Tensor,
) *tensor.Tensor {
	query := linear(queryInput, prefix+"_q")
	key := linear(keyValueInput, prefix+"_k")
	value := linear(keyValueInput, prefix+"_v")
	queryOrder := granite4AttentionOrder(windows, queryRows)
	keyOrder := granite4AttentionOrder(windows, keyRows)
	query = builder.GetRows(query, queryOrder)
	key = builder.GetRows(key, keyOrder)
	value = builder.GetRows(value, keyOrder)
	heads := hidden / 64
	query = builder.Reshape(query, 64, uint64(heads*windows), uint64(queryRows))
	key = builder.Reshape(key, 64, uint64(heads*windows), uint64(keyRows))
	value = builder.Reshape(value, 64, uint64(heads*windows), uint64(keyRows))
	output := builder.Attention(query, key, value, 1/8.0, false)
	output = builder.Reshape(output, uint64(hidden), uint64(windows*queryRows))
	inverse := make([]uint32, 0, windows*queryRows)
	for window := 0; window < windows; window++ {
		for row := 0; row < queryRows; row++ {
			inverse = append(inverse, uint32(row*windows+window))
		}
	}
	return linear(builder.GetRows(output, inverse), prefix+"_out")
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
