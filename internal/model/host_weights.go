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

// HostLayer: one dense layer dequantized to contiguous F32 values
type HostLayer struct {
	AttentionNorm               reference.Value
	AttentionNormBias           *reference.Value
	AttentionNorm2              *reference.Value
	AttentionNorm2Bias          *reference.Value
	AttentionQ                  reference.Value
	AttentionQB                 *reference.Value
	AttentionK                  reference.Value
	AttentionV                  reference.Value
	AttentionOutput             reference.Value
	AttentionQScale             *reference.Value
	AttentionKScale             *reference.Value
	AttentionVScale             *reference.Value
	AttentionOutputScale        *reference.Value
	AttentionSubNorm            *reference.Value
	AttentionQBias              *reference.Value
	AttentionKBias              *reference.Value
	AttentionVBias              *reference.Value
	AttentionOutputBias         *reference.Value
	AttentionQNorm              *reference.Value
	AttentionKNorm              *reference.Value
	AttentionQNormBias          *reference.Value
	AttentionKNormBias          *reference.Value
	AttentionPostNorm           *reference.Value
	AttentionPostNormBias       *reference.Value
	AttentionRelativeBias       *reference.Value
	AttentionOutputGate         *reference.Value
	AttentionSinks              *reference.Value
	RopeFactors                 *reference.Value
	FeedForwardNorm             reference.Value
	FeedForwardNormBias         *reference.Value
	FeedForwardExpertNorm       *reference.Value
	FeedForwardGate             reference.Value
	FeedForwardUp               reference.Value
	FeedForwardDown             reference.Value
	FeedForwardGateScale        *reference.Value
	FeedForwardUpScale          *reference.Value
	FeedForwardDownScale        *reference.Value
	FeedForwardSubNorm          *reference.Value
	FeedForwardGateBias         *reference.Value
	FeedForwardUpBias           *reference.Value
	FeedForwardDownBias         *reference.Value
	FeedForwardPostNorm         *reference.Value
	FeedForwardPostNormBias     *reference.Value
	FeedForwardPreNorm2         *reference.Value
	FeedForwardPostNorm1        *reference.Value
	FeedForwardPostNorm2        *reference.Value
	FeedForwardRouter           *reference.Value
	FeedForwardRouterBias       *reference.Value
	FeedForwardRouterScale      *reference.Value
	FeedForwardGateUpExperts    *reference.Value
	FeedForwardGateExperts      *reference.Value
	FeedForwardUpExperts        *reference.Value
	FeedForwardDownExperts      *reference.Value
	FeedForwardDownExpertsScale *reference.Value
	FeedForwardGateChunkExperts *reference.Value
	FeedForwardUpChunkExperts   *reference.Value
	FeedForwardDownChunkExperts *reference.Value
	FeedForwardExpertBias       *reference.Value
	FeedForwardSharedGate       *reference.Value
	FeedForwardSharedUp         *reference.Value
	FeedForwardSharedDown       *reference.Value
	FeedForwardSharedRouter     *reference.Value
	LayerOutputScale            *reference.Value
	PerLayerInputGate           *reference.Value
	PerLayerProjection          *reference.Value
	PerLayerPostNorm            *reference.Value
	ShortConvKernel             *reference.Value
	ShortConvInput              *reference.Value
	ShortConvOutput             *reference.Value
	AttentionKVAMQA             *reference.Value
	AttentionKVANorm            *reference.Value
	AttentionKVB                *reference.Value
	AttentionKB                 *reference.Value
	AttentionVB                 *reference.Value

	AttentionQKV      *reference.Value
	AttentionQKVBias  *reference.Value
	AttentionGate     *reference.Value
	SSMConv1D         *reference.Value
	SSMConv1DBias     *reference.Value
	SSMInput          *reference.Value
	SSMX              *reference.Value
	SSMTimeStepWeight *reference.Value
	SSMTimeStep       *reference.Value
	SSMA              *reference.Value
	SSMD              *reference.Value
	SSMBeta           *reference.Value
	SSMAlpha          *reference.Value
	SSMBetaAlpha      *reference.Value
	SSMNorm           *reference.Value
	SSMOutput         *reference.Value
}

// LoadHostTensor: reads and dequantizes one GGUF tensor
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

// LoadHostRows dequantizes selected rows from rank-2 table without
// materializing complete tensor
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

// ArgmaxDot: streams rank-2 output table and returns row with largest
// dot product against vector; bounds temporary memory by chunkRows
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

// DotRows: streams rank-2 output table and computes one logit per row
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
// before loading next layer
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
	}{}
	if info.FeedForwardUp.Name != "" && info.FeedForwardDown.Name != "" {
		items = append(items,
			struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.FeedForwardUp, info.FeedForwardUp},
			struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.FeedForwardDown, info.FeedForwardDown},
		)
	}
	if info.FeedForwardGate.Name != "" {
		items = append(items, struct {
			destination *reference.Value
			info        gguf.TensorInfo
		}{&result.FeedForwardGate, info.FeedForwardGate})
	}
	if info.AttentionNorm.Name != "" {
		items = append(items, struct {
			destination *reference.Value
			info        gguf.TensorInfo
		}{&result.AttentionNorm, info.AttentionNorm})
	}
	if info.FeedForwardNorm.Name != "" {
		items = append(items, struct {
			destination *reference.Value
			info        gguf.TensorInfo
		}{&result.FeedForwardNorm, info.FeedForwardNorm})
	}
	if info.Recurrent {
		if info.ShortConvKernel != nil && (info.ShortConvInput == nil || info.ShortConvOutput == nil) {
			return HostLayer{}, errors.New("host recurrent convolution catalog is incomplete")
		}
		if info.SSMInput != nil &&
			(info.SSMConv1D == nil || info.SSMConv1DBias == nil || info.SSMX == nil ||
				info.SSMTimeStepWeight == nil || info.SSMTimeStep == nil || info.SSMA == nil ||
				info.SSMD == nil || info.SSMOutput == nil) {
			return HostLayer{}, errors.New("host Mamba SSM catalog is incomplete")
		}
		if info.ShortConvKernel == nil && info.SSMInput == nil &&
			(info.AttentionQKV == nil || info.SSMConv1D == nil || info.SSMTimeStep == nil ||
				info.SSMA == nil || info.SSMNorm == nil || info.SSMOutput == nil ||
				(info.SSMBetaAlpha == nil && (info.SSMBeta == nil || info.SSMAlpha == nil))) {
			return HostLayer{}, errors.New("host recurrent layer catalog is incomplete")
		}
		optionalItems := []struct {
			destination **reference.Value
			info        *gguf.TensorInfo
		}{}
		if info.ShortConvKernel != nil {
			optionalItems = append(optionalItems,
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.ShortConvKernel, info.ShortConvKernel},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.ShortConvInput, info.ShortConvInput},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.ShortConvOutput, info.ShortConvOutput},
			)
		} else if info.SSMInput != nil {
			optionalItems = append(optionalItems,
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMInput, info.SSMInput},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMConv1D, info.SSMConv1D},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMConv1DBias, info.SSMConv1DBias},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMX, info.SSMX},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMTimeStepWeight, info.SSMTimeStepWeight},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMTimeStep, info.SSMTimeStep},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMA, info.SSMA},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMD, info.SSMD},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMOutput, info.SSMOutput},
			)
		} else {
			optionalItems = append(optionalItems,
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.AttentionQKV, info.AttentionQKV},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.AttentionGate, info.AttentionGate},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMConv1D, info.SSMConv1D},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMTimeStep, info.SSMTimeStep},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMA, info.SSMA},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMBeta, info.SSMBeta},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMAlpha, info.SSMAlpha},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMBetaAlpha, info.SSMBetaAlpha},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMNorm, info.SSMNorm},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMOutput, info.SSMOutput},
			)
		}
		for _, item := range optionalItems {
			if item.info == nil {
				continue
			}
			value, valueErr := LoadHostTensor(ctx, file, *item.info)
			if valueErr != nil {
				return HostLayer{}, valueErr
			}
			*item.destination = &value
		}
	} else {
		if info.AttentionOutput.Name == "" {
			// Attention-free layer.
		} else if info.AttentionQKV != nil {
			value, valueErr := LoadHostTensor(ctx, file, *info.AttentionQKV)
			if valueErr != nil {
				return HostLayer{}, valueErr
			}
			result.AttentionQKV = &value
		} else if info.AttentionKVAMQA != nil {
			items = append(items, struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.AttentionQ, info.AttentionQ})
		} else if info.AttentionQ.Name != "" {
			items = append(items, struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.AttentionQ, info.AttentionQ})
			if info.AttentionK.Name != "" {
				items = append(items, struct {
					destination *reference.Value
					info        gguf.TensorInfo
				}{&result.AttentionK, info.AttentionK})
			}
			if info.AttentionV.Name != "" {
				items = append(items, struct {
					destination *reference.Value
					info        gguf.TensorInfo
				}{&result.AttentionV, info.AttentionV})
			}
		}
		if info.AttentionOutput.Name != "" {
			items = append(items, struct {
				destination *reference.Value
				info        gguf.TensorInfo
			}{&result.AttentionOutput, info.AttentionOutput})
		}
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
	for _, item := range []struct {
		info        *gguf.TensorInfo
		destination **reference.Value
	}{
		{info.AttentionQB, &result.AttentionQB},
		{info.AttentionQScale, &result.AttentionQScale},
		{info.AttentionKScale, &result.AttentionKScale},
		{info.AttentionVScale, &result.AttentionVScale},
		{info.AttentionOutputScale, &result.AttentionOutputScale},
		{info.AttentionSubNorm, &result.AttentionSubNorm},
		{info.AttentionOutputGate, &result.AttentionOutputGate},
		{info.AttentionSinks, &result.AttentionSinks},
		{info.AttentionNormBias, &result.AttentionNormBias},
		{info.AttentionNorm2, &result.AttentionNorm2},
		{info.AttentionNorm2Bias, &result.AttentionNorm2Bias},
		{info.AttentionQKVBias, &result.AttentionQKVBias},
		{info.AttentionQNormBias, &result.AttentionQNormBias},
		{info.AttentionKNormBias, &result.AttentionKNormBias},
		{info.AttentionQBias, &result.AttentionQBias},
		{info.AttentionKBias, &result.AttentionKBias},
		{info.AttentionVBias, &result.AttentionVBias},
		{info.AttentionOutputBias, &result.AttentionOutputBias},
		{info.AttentionPostNormBias, &result.AttentionPostNormBias},
		{info.FeedForwardGateBias, &result.FeedForwardGateBias},
		{info.FeedForwardUpBias, &result.FeedForwardUpBias},
		{info.FeedForwardDownBias, &result.FeedForwardDownBias},
		{info.FeedForwardNormBias, &result.FeedForwardNormBias},
		{info.FeedForwardPostNormBias, &result.FeedForwardPostNormBias},
		{info.FeedForwardPreNorm2, &result.FeedForwardPreNorm2},
		{info.FeedForwardPostNorm1, &result.FeedForwardPostNorm1},
		{info.FeedForwardPostNorm2, &result.FeedForwardPostNorm2},
		{info.FeedForwardExpertNorm, &result.FeedForwardExpertNorm},
		{info.FeedForwardGateScale, &result.FeedForwardGateScale},
		{info.FeedForwardUpScale, &result.FeedForwardUpScale},
		{info.FeedForwardDownScale, &result.FeedForwardDownScale},
		{info.FeedForwardSubNorm, &result.FeedForwardSubNorm},
		{info.FeedForwardRouter, &result.FeedForwardRouter},
		{info.FeedForwardRouterScale, &result.FeedForwardRouterScale},
		{info.FeedForwardDownExpertsScale, &result.FeedForwardDownExpertsScale},
		{info.FeedForwardGateUpExperts, &result.FeedForwardGateUpExperts},
		{info.FeedForwardGateExperts, &result.FeedForwardGateExperts},
		{info.FeedForwardUpExperts, &result.FeedForwardUpExperts},
		{info.FeedForwardDownExperts, &result.FeedForwardDownExperts},
		{info.FeedForwardGateChunkExperts, &result.FeedForwardGateChunkExperts},
		{info.FeedForwardUpChunkExperts, &result.FeedForwardUpChunkExperts},
		{info.FeedForwardDownChunkExperts, &result.FeedForwardDownChunkExperts},
		{info.FeedForwardExpertBias, &result.FeedForwardExpertBias},
		{info.FeedForwardSharedGate, &result.FeedForwardSharedGate},
		{info.FeedForwardSharedUp, &result.FeedForwardSharedUp},
		{info.FeedForwardSharedDown, &result.FeedForwardSharedDown},
		{info.FeedForwardSharedRouter, &result.FeedForwardSharedRouter},
		{info.LayerOutputScale, &result.LayerOutputScale},
		{info.PerLayerInputGate, &result.PerLayerInputGate},
		{info.PerLayerProjection, &result.PerLayerProjection},
		{info.PerLayerPostNorm, &result.PerLayerPostNorm},
		{info.AttentionKVAMQA, &result.AttentionKVAMQA},
		{info.AttentionKVANorm, &result.AttentionKVANorm},
		{info.AttentionKVB, &result.AttentionKVB},
		{info.AttentionKB, &result.AttentionKB},
		{info.AttentionVB, &result.AttentionVB},
	} {
		if item.info == nil {
			continue
		}
		value, err := LoadHostTensor(ctx, file, *item.info)
		if err != nil {
			return HostLayer{}, err
		}
		*item.destination = &value
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
	if info.RopeFactors != nil {
		value, err := LoadHostTensor(ctx, file, *info.RopeFactors)
		if err != nil {
			return HostLayer{}, err
		}
		result.RopeFactors = &value
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
// feeds without copying underlying dequantized slices
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
	result := LayerGraphWeights{}
	if layer.FeedForwardUp.Shape.Rank != 0 && layer.FeedForwardDown.Shape.Rank != 0 {
		result.FeedForwardUp = input("ffn_up.weight", layer.FeedForwardUp)
		result.FeedForwardDown = input("ffn_down.weight", layer.FeedForwardDown)
	}
	if layer.FeedForwardGate.Shape.Rank != 0 {
		result.FeedForwardGate = input("ffn_gate.weight", layer.FeedForwardGate)
	}
	if layer.AttentionNorm.Shape.Rank != 0 {
		result.AttentionNorm = input("attn_norm.weight", layer.AttentionNorm)
	}
	if layer.AttentionNormBias != nil {
		result.AttentionNormBias = input("attn_norm.bias", *layer.AttentionNormBias)
	}
	if layer.AttentionNorm2 != nil {
		result.AttentionNorm2 = input("attn_norm_2.weight", *layer.AttentionNorm2)
	}
	if layer.AttentionNorm2Bias != nil {
		result.AttentionNorm2Bias = input("attn_norm_2.bias", *layer.AttentionNorm2Bias)
	}
	if layer.FeedForwardNorm.Shape.Rank != 0 {
		result.FeedForwardNorm = input("ffn_norm.weight", layer.FeedForwardNorm)
	}
	if layer.FeedForwardNormBias != nil {
		result.FeedForwardNormBias = input("ffn_norm.bias", *layer.FeedForwardNormBias)
	}
	if layer.SSMInput != nil {
		result.SSMInput = input("ssm_in.weight", *layer.SSMInput)
		result.SSMConv1D = input("ssm_conv1d.weight", *layer.SSMConv1D)
		result.SSMConv1DBias = input("ssm_conv1d.bias", *layer.SSMConv1DBias)
		result.SSMX = input("ssm_x.weight", *layer.SSMX)
		result.SSMTimeStepWeight = input("ssm_dt.weight", *layer.SSMTimeStepWeight)
		result.SSMTimeStep = input("ssm_dt.bias", *layer.SSMTimeStep)
		result.SSMA = input("ssm_a", *layer.SSMA)
		result.SSMD = input("ssm_d", *layer.SSMD)
		result.SSMOutput = input("ssm_out.weight", *layer.SSMOutput)
	} else if layer.AttentionQKV != nil {
		result.AttentionQKV = input("attn_qkv.weight", *layer.AttentionQKV)
		if layer.SSMConv1D != nil {
			if layer.AttentionGate != nil {
				result.AttentionGate = input("attn_gate.weight", *layer.AttentionGate)
			}
			result.SSMConv1D = input("ssm_conv1d.weight", *layer.SSMConv1D)
			result.SSMTimeStep = input("ssm_dt.bias", *layer.SSMTimeStep)
			result.SSMA = input("ssm_a", *layer.SSMA)
			if layer.SSMBeta != nil {
				result.SSMBeta = input("ssm_beta.weight", *layer.SSMBeta)
			}
			if layer.SSMAlpha != nil {
				result.SSMAlpha = input("ssm_alpha.weight", *layer.SSMAlpha)
			}
			if layer.SSMBetaAlpha != nil {
				result.SSMBetaAlpha = input("ssm_ba.weight", *layer.SSMBetaAlpha)
			}
			result.SSMNorm = input("ssm_norm.weight", *layer.SSMNorm)
			result.SSMOutput = input("ssm_out.weight", *layer.SSMOutput)
		} else {
			result.AttentionOutput = input("attn_output.weight", layer.AttentionOutput)
		}
	} else if layer.ShortConvKernel != nil {
		result.ShortConvKernel = input("shortconv.conv.weight", *layer.ShortConvKernel)
		result.ShortConvInput = input("shortconv.in_proj.weight", *layer.ShortConvInput)
		result.ShortConvOutput = input("shortconv.out_proj.weight", *layer.ShortConvOutput)
	} else if layer.AttentionOutput.Shape.Rank == 0 {
		// Attention-free.
	} else if layer.AttentionKVAMQA != nil {
		result.AttentionQ = input("attn_q.weight", layer.AttentionQ)
		result.AttentionOutput = input("attn_output.weight", layer.AttentionOutput)
	} else if layer.AttentionQ.Shape.Rank != 0 {
		result.AttentionQ = input("attn_q.weight", layer.AttentionQ)
		if layer.AttentionK.Shape.Rank != 0 {
			result.AttentionK = input("attn_k.weight", layer.AttentionK)
		}
		if layer.AttentionV.Shape.Rank != 0 {
			result.AttentionV = input("attn_v.weight", layer.AttentionV)
		}
		result.AttentionOutput = input("attn_output.weight", layer.AttentionOutput)
	} else {
		result.AttentionOutput = input("attn_output.weight", layer.AttentionOutput)
	}
	if layer.AttentionQNorm != nil {
		result.AttentionQNorm = input("attn_q_norm.weight", *layer.AttentionQNorm)
	}
	for _, item := range []struct {
		name        string
		value       *reference.Value
		destination **tensor.Tensor
	}{
		{"attn_q_b.weight", layer.AttentionQB, &result.AttentionQB},
		{"attn_q.scale", layer.AttentionQScale, &result.AttentionQScale},
		{"attn_k.scale", layer.AttentionKScale, &result.AttentionKScale},
		{"attn_v.scale", layer.AttentionVScale, &result.AttentionVScale},
		{"attn_output.scale", layer.AttentionOutputScale, &result.AttentionOutputScale},
		{"attn_sub_norm.weight", layer.AttentionSubNorm, &result.AttentionSubNorm},
		{"attn_gate.weight", layer.AttentionOutputGate, &result.AttentionOutputGate},
		{"attn_sinks.weight", layer.AttentionSinks, &result.AttentionSinks},
		{"attn_qkv.bias", layer.AttentionQKVBias, &result.AttentionQKVBias},
		{"attn_q_norm.bias", layer.AttentionQNormBias, &result.AttentionQNormBias},
		{"attn_k_norm.bias", layer.AttentionKNormBias, &result.AttentionKNormBias},
		{"attn_q.bias", layer.AttentionQBias, &result.AttentionQBias},
		{"attn_k.bias", layer.AttentionKBias, &result.AttentionKBias},
		{"attn_v.bias", layer.AttentionVBias, &result.AttentionVBias},
		{"attn_output.bias", layer.AttentionOutputBias, &result.AttentionOutputBias},
		{"post_attention_norm.bias", layer.AttentionPostNormBias, &result.AttentionPostNormBias},
		{"ffn_gate.bias", layer.FeedForwardGateBias, &result.FeedForwardGateBias},
		{"ffn_up.bias", layer.FeedForwardUpBias, &result.FeedForwardUpBias},
		{"ffn_down.bias", layer.FeedForwardDownBias, &result.FeedForwardDownBias},
		{"post_ffw_norm.bias", layer.FeedForwardPostNormBias, &result.FeedForwardPostNormBias},
		{"pre_ffw_norm_2.weight", layer.FeedForwardPreNorm2, &result.FeedForwardPreNorm2},
		{"post_ffw_norm_1.weight", layer.FeedForwardPostNorm1, &result.FeedForwardPostNorm1},
		{"post_ffw_norm_2.weight", layer.FeedForwardPostNorm2, &result.FeedForwardPostNorm2},
		{"ffn_norm_exps.weight", layer.FeedForwardExpertNorm, &result.FeedForwardExpertNorm},
		{"ffn_gate.scale", layer.FeedForwardGateScale, &result.FeedForwardGateScale},
		{"ffn_up.scale", layer.FeedForwardUpScale, &result.FeedForwardUpScale},
		{"ffn_down.scale", layer.FeedForwardDownScale, &result.FeedForwardDownScale},
		{"ffn_sub_norm.weight", layer.FeedForwardSubNorm, &result.FeedForwardSubNorm},
		{"ffn_gate_inp.weight", layer.FeedForwardRouter, &result.FeedForwardRouter},
		{"ffn_gate_inp.bias", layer.FeedForwardRouterBias, &result.FeedForwardRouterBias},
		{"ffn_gate_inp.scale", layer.FeedForwardRouterScale, &result.FeedForwardRouterScale},
		{"ffn_down_exps.scale", layer.FeedForwardDownExpertsScale, &result.FeedForwardDownExpertsScale},
		{"ffn_gate_up_exps.weight", layer.FeedForwardGateUpExperts, &result.FeedForwardGateUpExperts},
		{"ffn_gate_exps.weight", layer.FeedForwardGateExperts, &result.FeedForwardGateExperts},
		{"ffn_up_exps.weight", layer.FeedForwardUpExperts, &result.FeedForwardUpExperts},
		{"ffn_down_exps.weight", layer.FeedForwardDownExperts, &result.FeedForwardDownExperts},
		{"ffn_gate_chexps.weight", layer.FeedForwardGateChunkExperts, &result.FeedForwardGateChunkExperts},
		{"ffn_up_chexps.weight", layer.FeedForwardUpChunkExperts, &result.FeedForwardUpChunkExperts},
		{"ffn_down_chexps.weight", layer.FeedForwardDownChunkExperts, &result.FeedForwardDownChunkExperts},
		{"exp_probs_b.bias", layer.FeedForwardExpertBias, &result.FeedForwardExpertBias},
		{"ffn_gate_shexp.weight", layer.FeedForwardSharedGate, &result.FeedForwardSharedGate},
		{"ffn_up_shexp.weight", layer.FeedForwardSharedUp, &result.FeedForwardSharedUp},
		{"ffn_down_shexp.weight", layer.FeedForwardSharedDown, &result.FeedForwardSharedDown},
		{"ffn_gate_inp_shexp.weight", layer.FeedForwardSharedRouter, &result.FeedForwardSharedRouter},
		{"layer_output_scale.weight", layer.LayerOutputScale, &result.LayerOutputScale},
		{"per_layer_inp_gate.weight", layer.PerLayerInputGate, &result.PerLayerInputGate},
		{"per_layer_proj.weight", layer.PerLayerProjection, &result.PerLayerProjection},
		{"per_layer_post_norm.weight", layer.PerLayerPostNorm, &result.PerLayerPostNorm},
		{"attn_kv_a_mqa.weight", layer.AttentionKVAMQA, &result.AttentionKVAMQA},
		{"attn_kv_a_norm.weight", layer.AttentionKVANorm, &result.AttentionKVANorm},
		{"attn_kv_b.weight", layer.AttentionKVB, &result.AttentionKVB},
		{"attn_k_b.weight", layer.AttentionKB, &result.AttentionKB},
		{"attn_v_b.weight", layer.AttentionVB, &result.AttentionVB},
	} {
		if item.value != nil {
			*item.destination = input(item.name, *item.value)
		}
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
	if layer.RopeFactors != nil {
		result.RopeFactors = input("rope_freqs.weight", *layer.RopeFactors)
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
