package model

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/quant"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

// HostLayer is one dense layer dequantized to contiguous F32 values.
type HostLayer struct {
	AttentionNorm         reference.Value
	AttentionQ            reference.Value
	AttentionK            reference.Value
	AttentionV            reference.Value
	AttentionOutput       reference.Value
	AttentionQNorm        *reference.Value
	AttentionKNorm        *reference.Value
	AttentionPostNorm     *reference.Value
	AttentionRelativeBias *reference.Value
	FeedForwardNorm       reference.Value
	FeedForwardGate       reference.Value
	FeedForwardUp         reference.Value
	FeedForwardDown       reference.Value
	FeedForwardPostNorm   *reference.Value

	AttentionQKV  *reference.Value
	AttentionGate *reference.Value
	SSMConv1D     *reference.Value
	SSMTimeStep   *reference.Value
	SSMA          *reference.Value
	SSMBeta       *reference.Value
	SSMAlpha      *reference.Value
	SSMNorm       *reference.Value
	SSMOutput     *reference.Value
}

// LoadHostTensor reads and dequantizes one GGUF tensor.
func LoadHostTensor(ctx context.Context, file *gguf.File, info gguf.TensorInfo) (reference.Value, error) {
	if file == nil {
		return reference.Value{}, errors.New("host tensor: GGUF file is nil")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	shapeDimensions := make([]uint64, info.Dimensions)
	copy(shapeDimensions, info.Shape[:info.Dimensions])
	shape, err := tensor.NewShape(shapeDimensions...)
	if err != nil {
		return reference.Value{}, fmt.Errorf("host tensor %q shape: %w", info.Name, err)
	}
	elements, err := shape.Elements()
	if err != nil {
		return reference.Value{}, fmt.Errorf("host tensor %q elements: %w", info.Name, err)
	}
	if info.Size > uint64(maxInt()) {
		return reference.Value{}, fmt.Errorf("host tensor %q storage exceeds addressable memory", info.Name)
	}
	storage := make([]byte, int(info.Size))
	if err := file.ReadTensorData(info, storage); err != nil {
		return reference.Value{}, fmt.Errorf("read host tensor %q: %w", info.Name, err)
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	data, err := quant.Dequantize(info.Type, storage, elements)
	if err != nil {
		return reference.Value{}, fmt.Errorf("dequantize host tensor %q: %w", info.Name, err)
	}
	return reference.Value{Shape: shape, Data: data}, nil
}

// LoadHostRows dequantizes selected rows from a rank-2 table without
// materializing the complete tensor.
func LoadHostRows(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
	rows []uint32,
) (reference.Value, error) {
	if file == nil {
		return reference.Value{}, errors.New("host rows: GGUF file is nil")
	}
	if info.Dimensions != 2 {
		return reference.Value{}, fmt.Errorf("host rows: tensor %q must have rank 2", info.Name)
	}
	if len(rows) == 0 {
		return reference.Value{}, errors.New("host rows: row list is empty")
	}
	traits, ok := info.Type.Traits()
	if !ok {
		return reference.Value{}, fmt.Errorf("host rows: tensor %q has unknown type %d", info.Name, info.Type)
	}
	width := info.Shape[0]
	if width%traits.BlockSize != 0 {
		return reference.Value{}, fmt.Errorf("host rows: tensor %q row width is not block aligned", info.Name)
	}
	rowBlocks := width / traits.BlockSize
	if rowBlocks > math.MaxUint64/traits.TypeSize {
		return reference.Value{}, errors.New("host rows: row byte size overflows")
	}
	rowBytes := rowBlocks * traits.TypeSize
	if info.Shape[1] > math.MaxUint64/rowBytes || info.Shape[1]*rowBytes != info.Size {
		return reference.Value{}, fmt.Errorf("host rows: tensor %q storage is inconsistent with its shape", info.Name)
	}
	if rowBytes > uint64(maxInt()) || width > uint64(maxInt()) ||
		uint64(len(rows)) > math.MaxUint64/width ||
		uint64(len(rows))*width > uint64(maxInt()) {
		return reference.Value{}, errors.New("host rows: result exceeds addressable memory")
	}
	storage := make([]byte, int(rowBytes))
	output := make([]float32, int(width)*len(rows))
	for outputRow, row := range rows {
		if err := ctx.Err(); err != nil {
			return reference.Value{}, err
		}
		if uint64(row) >= info.Shape[1] {
			return reference.Value{}, fmt.Errorf("host rows: row %d exceeds tensor row count %d", row, info.Shape[1])
		}
		offset := uint64(row) * rowBytes
		if err := file.ReadTensorRange(info, offset, storage); err != nil {
			return reference.Value{}, fmt.Errorf("read host row %d from %q: %w", row, info.Name, err)
		}
		values, err := quant.Dequantize(info.Type, storage, width)
		if err != nil {
			return reference.Value{}, fmt.Errorf("dequantize host row %d from %q: %w", row, info.Name, err)
		}
		copy(output[outputRow*int(width):(outputRow+1)*int(width)], values)
	}
	shape, err := tensor.NewShape(width, uint64(len(rows)))
	if err != nil {
		return reference.Value{}, err
	}
	return reference.Value{Shape: shape, Data: output}, nil
}

// ArgmaxDot streams a rank-2 output table and returns the row with the largest
// dot product against vector. It bounds temporary memory by chunkRows.
func ArgmaxDot(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
	vector []float32,
	chunkRows uint32,
) (uint32, float32, error) {
	scores, err := DotRows(ctx, file, info, vector, chunkRows)
	if err != nil {
		return 0, 0, err
	}
	bestRow := uint32(0)
	bestScore := float32(math.Inf(-1))
	for row, score := range scores {
		if score > bestScore {
			bestScore = score
			bestRow = uint32(row)
		}
	}
	return bestRow, bestScore, nil
}

// DotRows streams a rank-2 output table and computes one logit per row.
func DotRows(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
	vector []float32,
	chunkRows uint32,
) ([]float32, error) {
	if file == nil {
		return nil, errors.New("dot rows: GGUF file is nil")
	}
	if info.Dimensions != 2 {
		return nil, fmt.Errorf("dot rows: tensor %q must have rank 2", info.Name)
	}
	width := info.Shape[0]
	totalRows := info.Shape[1]
	if uint64(len(vector)) != width {
		return nil, fmt.Errorf("dot rows: vector has %d values, need %d", len(vector), width)
	}
	if totalRows == 0 || totalRows > math.MaxUint32 {
		return nil, errors.New("dot rows: row count is invalid")
	}
	if chunkRows == 0 {
		chunkRows = 1024
	}
	traits, ok := info.Type.Traits()
	if !ok || width%traits.BlockSize != 0 {
		return nil, fmt.Errorf("dot rows: tensor %q has unsupported row layout", info.Name)
	}
	rowBytes := width / traits.BlockSize * traits.TypeSize
	if totalRows > math.MaxUint64/rowBytes || totalRows*rowBytes != info.Size {
		return nil, fmt.Errorf("dot rows: tensor %q storage is inconsistent with its shape", info.Name)
	}

	scores := make([]float32, int(totalRows))
	for first := uint64(0); first < totalRows; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count := uint64(chunkRows)
		if remaining := totalRows - first; count > remaining {
			count = remaining
		}
		storageBytes := count * rowBytes
		if storageBytes > uint64(maxInt()) || count > math.MaxUint64/width || count*width > uint64(maxInt()) {
			return nil, errors.New("dot rows: chunk exceeds addressable memory")
		}
		storage := make([]byte, int(storageBytes))
		if err := file.ReadTensorRange(info, first*rowBytes, storage); err != nil {
			return nil, fmt.Errorf("read output rows at %d: %w", first, err)
		}
		values, err := quant.Dequantize(info.Type, storage, count*width)
		if err != nil {
			return nil, fmt.Errorf("dequantize output rows at %d: %w", first, err)
		}
		for row := uint64(0); row < count; row++ {
			var dot float64
			offset := int(row * width)
			for column, input := range vector {
				dot += float64(values[offset+column]) * float64(input)
			}
			scores[first+row] = float32(dot)
		}
		first += count
	}
	return scores, nil
}

// LoadHostLayer materializes only one layer, allowing callers to release it
// before loading the next layer.
func LoadHostLayer(
	ctx context.Context,
	file *gguf.File,
	info LayerWeights,
) (HostLayer, error) {
	var result HostLayer
	load := func(destination *reference.Value, tensorInfo gguf.TensorInfo) error {
		value, err := LoadHostTensor(ctx, file, tensorInfo)
		if err != nil {
			return err
		}
		*destination = value
		return nil
	}
	items := []struct {
		destination *reference.Value
		info        gguf.TensorInfo
	}{
		{&result.AttentionNorm, info.AttentionNorm},
		{&result.FeedForwardNorm, info.FeedForwardNorm},
		{&result.FeedForwardGate, info.FeedForwardGate},
		{&result.FeedForwardUp, info.FeedForwardUp},
		{&result.FeedForwardDown, info.FeedForwardDown},
	}
	if info.Recurrent {
		optionalItems := []struct {
			destination **reference.Value
			info        *gguf.TensorInfo
		}{
			{&result.AttentionQKV, info.AttentionQKV},
			{&result.AttentionGate, info.AttentionGate},
			{&result.SSMConv1D, info.SSMConv1D},
			{&result.SSMTimeStep, info.SSMTimeStep},
			{&result.SSMA, info.SSMA},
			{&result.SSMBeta, info.SSMBeta},
			{&result.SSMAlpha, info.SSMAlpha},
			{&result.SSMNorm, info.SSMNorm},
			{&result.SSMOutput, info.SSMOutput},
		}
		for _, item := range optionalItems {
			if item.info == nil {
				return HostLayer{}, errors.New("host recurrent layer catalog is incomplete")
			}
			value, valueErr := LoadHostTensor(ctx, file, *item.info)
			if valueErr != nil {
				return HostLayer{}, valueErr
			}
			*item.destination = &value
		}
	} else {
		items = append(items,
			struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.AttentionQ, info.AttentionQ},
			struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.AttentionK, info.AttentionK},
			struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.AttentionV, info.AttentionV},
			struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.AttentionOutput, info.AttentionOutput},
		)
	}
	for _, item := range items {
		if err := load(item.destination, item.info); err != nil {
			return HostLayer{}, err
		}
	}
	if info.AttentionQNorm != nil {
		value, err := LoadHostTensor(ctx, file, *info.AttentionQNorm)
		if err != nil {
			return HostLayer{}, err
		}
		result.AttentionQNorm = &value
	}
	if info.AttentionKNorm != nil {
		value, err := LoadHostTensor(ctx, file, *info.AttentionKNorm)
		if err != nil {
			return HostLayer{}, err
		}
		result.AttentionKNorm = &value
	}
	if info.AttentionPostNorm != nil {
		value, err := LoadHostTensor(ctx, file, *info.AttentionPostNorm)
		if err != nil {
			return HostLayer{}, err
		}
		result.AttentionPostNorm = &value
	}
	if info.AttentionRelativeBias != nil {
		value, err := LoadHostTensor(ctx, file, *info.AttentionRelativeBias)
		if err != nil {
			return HostLayer{}, err
		}
		result.AttentionRelativeBias = &value
	}
	if info.FeedForwardPostNorm != nil {
		value, err := LoadHostTensor(ctx, file, *info.FeedForwardPostNorm)
		if err != nil {
			return HostLayer{}, err
		}
		result.FeedForwardPostNorm = &value
	}
	return result, nil
}

// GraphInputs creates graph input nodes and their reference/CUDA executor
// feeds without copying the underlying dequantized slices.
func (layer *HostLayer) GraphInputs(
	builder *tensor.Builder,
	prefix string,
) (LayerGraphWeights, map[*tensor.Tensor]reference.Value, error) {
	if layer == nil {
		return LayerGraphWeights{}, nil, errors.New("host layer is nil")
	}
	if builder == nil {
		return LayerGraphWeights{}, nil, errors.New("host layer graph builder is nil")
	}
	feeds := make(map[*tensor.Tensor]reference.Value, 11)
	input := func(name string, value reference.Value) *tensor.Tensor {
		node := builder.Input(prefix+name, dtype.F32, value.Shape)
		if node != nil {
			feeds[node] = value
		}
		return node
	}
	result := LayerGraphWeights{
		AttentionNorm:   input("attn_norm.weight", layer.AttentionNorm),
		FeedForwardNorm: input("ffn_norm.weight", layer.FeedForwardNorm),
		FeedForwardGate: input("ffn_gate.weight", layer.FeedForwardGate),
		FeedForwardUp:   input("ffn_up.weight", layer.FeedForwardUp),
		FeedForwardDown: input("ffn_down.weight", layer.FeedForwardDown),
	}
	if layer.AttentionQKV != nil {
		result.AttentionQKV = input("attn_qkv.weight", *layer.AttentionQKV)
		result.AttentionGate = input("attn_gate.weight", *layer.AttentionGate)
		result.SSMConv1D = input("ssm_conv1d.weight", *layer.SSMConv1D)
		result.SSMTimeStep = input("ssm_dt.bias", *layer.SSMTimeStep)
		result.SSMA = input("ssm_a", *layer.SSMA)
		result.SSMBeta = input("ssm_beta.weight", *layer.SSMBeta)
		result.SSMAlpha = input("ssm_alpha.weight", *layer.SSMAlpha)
		result.SSMNorm = input("ssm_norm.weight", *layer.SSMNorm)
		result.SSMOutput = input("ssm_out.weight", *layer.SSMOutput)
	} else {
		result.AttentionQ = input("attn_q.weight", layer.AttentionQ)
		result.AttentionK = input("attn_k.weight", layer.AttentionK)
		result.AttentionV = input("attn_v.weight", layer.AttentionV)
		result.AttentionOutput = input("attn_output.weight", layer.AttentionOutput)
	}
	if layer.AttentionQNorm != nil {
		result.AttentionQNorm = input("attn_q_norm.weight", *layer.AttentionQNorm)
	}
	if layer.AttentionKNorm != nil {
		result.AttentionKNorm = input("attn_k_norm.weight", *layer.AttentionKNorm)
	}
	if layer.AttentionPostNorm != nil {
		result.AttentionPostNorm = input("post_attention_norm.weight", *layer.AttentionPostNorm)
	}
	if layer.AttentionRelativeBias != nil {
		result.AttentionRelativeBias = input("attn_rel_b.weight", *layer.AttentionRelativeBias)
	}
	if layer.FeedForwardPostNorm != nil {
		result.FeedForwardPostNorm = input("post_ffw_norm.weight", *layer.FeedForwardPostNorm)
	}
	if err := builder.Err(); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	return result, feeds, nil
}

func maxInt() int {
	return int(^uint(0) >> 1)
}
