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
	CrossAttentionNorm          *reference.Value
	CrossAttentionQ             *reference.Value
	CrossAttentionK             *reference.Value
	CrossAttentionV             *reference.Value
	CrossAttentionOutput        *reference.Value
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
	FeedForwardActivationScale  *reference.Value
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
	FeedForwardLatentDown       *reference.Value
	FeedForwardLatentUp         *reference.Value
	FeedForwardSharedGate       *reference.Value
	FeedForwardSharedUp         *reference.Value
	FeedForwardSharedDown       *reference.Value
	FeedForwardSharedRouter     *reference.Value
	LayerOutputScale            *reference.Value
	PerLayerInputGate           *reference.Value
	PerLayerProjection          *reference.Value
	PerLayerPostNorm            *reference.Value
	AltUpCorrectCoefficient     *reference.Value
	AltUpCorrectScale           *reference.Value
	AltUpPredictCoefficient     *reference.Value
	AltUpRouter                 *reference.Value
	AltUpRouterNorm             *reference.Value
	LaurelLeft                  *reference.Value
	LaurelRight                 *reference.Value
	LaurelPostNorm              *reference.Value
	ShortConvKernel             *reference.Value
	ShortConvInput              *reference.Value
	ShortConvOutput             *reference.Value
	AttentionKVAMQA             *reference.Value
	AttentionKVANorm            *reference.Value
	AttentionKVB                *reference.Value
	AttentionKB                 *reference.Value
	AttentionVB                 *reference.Value
	IndexerKNorm                *reference.Value
	IndexerKNormBias            *reference.Value
	IndexerProjection           *reference.Value
	IndexerAttentionK           *reference.Value
	IndexerAttentionQB          *reference.Value
	AttentionOutputA            *reference.Value
	AttentionCompressorKV       *reference.Value
	AttentionCompressorGate     *reference.Value
	AttentionCompressorAPE      *reference.Value
	AttentionCompressorNorm     *reference.Value
	IndexerCompressorKV         *reference.Value
	IndexerCompressorGate       *reference.Value
	IndexerCompressorAPE        *reference.Value
	IndexerCompressorNorm       *reference.Value
	HyperAttentionFN            *reference.Value
	HyperAttentionBase          *reference.Value
	HyperAttentionScale         *reference.Value
	HyperFeedForwardFN          *reference.Value
	HyperFeedForwardBase        *reference.Value
	HyperFeedForwardScale       *reference.Value
	HyperHeadFN                 *reference.Value
	HyperHeadBase               *reference.Value
	HyperHeadScale              *reference.Value
	FeedForwardHashExperts      *reference.Value

	AttentionQKV         *reference.Value
	AttentionQKVBias     *reference.Value
	AttentionGate        *reference.Value
	SSMConv1D            *reference.Value
	SSMConv1DBias        *reference.Value
	SSMInput             *reference.Value
	SSMX                 *reference.Value
	SSMTimeStepWeight    *reference.Value
	SSMTimeStep          *reference.Value
	SSMTimeStepNorm      *reference.Value
	SSMA                 *reference.Value
	SSMD                 *reference.Value
	SSMBNorm             *reference.Value
	SSMCNorm             *reference.Value
	SSMBeta              *reference.Value
	SSMAlpha             *reference.Value
	SSMBetaAlpha         *reference.Value
	SSMNorm              *reference.Value
	SSMOutput            *reference.Value
	SSMQueryConv         *reference.Value
	SSMKeyConv           *reference.Value
	SSMValueConv         *reference.Value
	SSMForgetA           *reference.Value
	SSMForgetB           *reference.Value
	SSMOutputGateA       *reference.Value
	SSMOutputGateB       *reference.Value
	TimeMixW1            *reference.Value
	TimeMixW2            *reference.Value
	TimeMixW0            *reference.Value
	TimeMixA0            *reference.Value
	TimeMixA1            *reference.Value
	TimeMixA2            *reference.Value
	TimeMixV0            *reference.Value
	TimeMixV1            *reference.Value
	TimeMixV2            *reference.Value
	TimeMixG1            *reference.Value
	TimeMixG2            *reference.Value
	TimeMixKK            *reference.Value
	TimeMixKA            *reference.Value
	TimeMixRK            *reference.Value
	TimeMixLerpX         *reference.Value
	TimeMixLerpFused     *reference.Value
	TimeMixLerpW         *reference.Value
	TimeMixLerpK         *reference.Value
	TimeMixLerpV         *reference.Value
	TimeMixLerpR         *reference.Value
	TimeMixLerpG         *reference.Value
	TimeMixFirst         *reference.Value
	TimeMixDecay         *reference.Value
	TimeMixDecayW1       *reference.Value
	TimeMixDecayW2       *reference.Value
	TimeMixKey           *reference.Value
	TimeMixValue         *reference.Value
	TimeMixReceptance    *reference.Value
	TimeMixGate          *reference.Value
	TimeMixLN            *reference.Value
	TimeMixLNBias        *reference.Value
	TimeMixOutput        *reference.Value
	ChannelMixLerpK      *reference.Value
	ChannelMixLerpR      *reference.Value
	ChannelMixKey        *reference.Value
	ChannelMixValue      *reference.Value
	ChannelMixReceptance *reference.Value
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
	if info.Size > uint64(math.MaxInt) {
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
	if rowBytes > uint64(math.MaxInt) || width > uint64(math.MaxInt) ||
		uint64(len(rows)) > math.MaxUint64/width ||
		uint64(len(rows))*width > uint64(math.MaxInt) {
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
		if storageBytes > uint64(math.MaxInt) || count > math.MaxUint64/width || count*width > uint64(math.MaxInt) {
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
	if !info.Recurrent && info.SSMInput != nil {
		if info.SSMConv1D == nil || info.SSMTimeStep == nil || info.SSMA == nil ||
			info.SSMD == nil || info.SSMOutput == nil {
			return HostLayer{}, errors.New("host hybrid SSM catalog is incomplete")
		}
		for _, item := range []struct {
			destination **reference.Value
			info        *gguf.TensorInfo
		}{
			{&result.SSMInput, info.SSMInput}, {&result.SSMConv1D, info.SSMConv1D},
			{&result.SSMConv1DBias, info.SSMConv1DBias}, {&result.SSMTimeStep, info.SSMTimeStep},
			{&result.SSMA, info.SSMA}, {&result.SSMD, info.SSMD},
			{&result.SSMNorm, info.SSMNorm}, {&result.SSMOutput, info.SSMOutput},
		} {
			if item.info == nil {
				continue
			}
			value, valueErr := LoadHostTensor(ctx, file, *item.info)
			if valueErr != nil {
				return HostLayer{}, valueErr
			}
			*item.destination = &value
		}
	}
	if info.Recurrent {
		kimi := info.SSMQueryConv != nil
		rwkv := info.TimeMixW1 != nil
		if info.ShortConvKernel != nil && (info.ShortConvInput == nil || info.ShortConvOutput == nil) {
			return HostLayer{}, errors.New("host recurrent convolution catalog is incomplete")
		}
		if info.SSMInput != nil &&
			(info.SSMConv1D == nil || info.SSMTimeStep == nil ||
				info.SSMA == nil || info.SSMD == nil || info.SSMOutput == nil ||
				(info.SSMX != nil && info.SSMTimeStepWeight == nil) ||
				(info.SSMTimeStepNorm != nil && (info.SSMBNorm == nil || info.SSMCNorm == nil)) ||
				(info.SSMX == nil && info.SSMNorm == nil)) {
			return HostLayer{}, errors.New("host Mamba SSM catalog is incomplete")
		}
		if !kimi && !rwkv && info.ShortConvKernel == nil && info.SSMInput == nil &&
			(info.AttentionQKV == nil || info.SSMConv1D == nil || info.SSMTimeStep == nil ||
				info.SSMA == nil || info.SSMNorm == nil || info.SSMOutput == nil ||
				(info.SSMBetaAlpha == nil && (info.SSMBeta == nil || info.SSMAlpha == nil))) {
			return HostLayer{}, errors.New("host recurrent layer catalog is incomplete")
		}
		optionalItems := []struct {
			destination **reference.Value
			info        *gguf.TensorInfo
		}{}
		if rwkv {
			for _, item := range []struct {
				destination **reference.Value
				info        *gguf.TensorInfo
			}{
				{&result.TimeMixW1, info.TimeMixW1},
				{&result.TimeMixW2, info.TimeMixW2},
				{&result.TimeMixW0, info.TimeMixW0},
				{&result.TimeMixA0, info.TimeMixA0},
				{&result.TimeMixA1, info.TimeMixA1},
				{&result.TimeMixA2, info.TimeMixA2},
				{&result.TimeMixV0, info.TimeMixV0},
				{&result.TimeMixV1, info.TimeMixV1},
				{&result.TimeMixV2, info.TimeMixV2},
				{&result.TimeMixG1, info.TimeMixG1},
				{&result.TimeMixG2, info.TimeMixG2},
				{&result.TimeMixKK, info.TimeMixKK},
				{&result.TimeMixKA, info.TimeMixKA},
				{&result.TimeMixRK, info.TimeMixRK},
				{&result.TimeMixLerpX, info.TimeMixLerpX},
				{&result.TimeMixLerpFused, info.TimeMixLerpFused},
				{&result.TimeMixLerpW, info.TimeMixLerpW},
				{&result.TimeMixLerpK, info.TimeMixLerpK},
				{&result.TimeMixLerpV, info.TimeMixLerpV},
				{&result.TimeMixLerpR, info.TimeMixLerpR},
				{&result.TimeMixLerpG, info.TimeMixLerpG},
				{&result.TimeMixFirst, info.TimeMixFirst},
				{&result.TimeMixDecay, info.TimeMixDecay},
				{&result.TimeMixDecayW1, info.TimeMixDecayW1},
				{&result.TimeMixDecayW2, info.TimeMixDecayW2},
				{&result.TimeMixKey, info.TimeMixKey},
				{&result.TimeMixValue, info.TimeMixValue},
				{&result.TimeMixReceptance, info.TimeMixReceptance},
				{&result.TimeMixGate, info.TimeMixGate},
				{&result.TimeMixLN, info.TimeMixLN},
				{&result.TimeMixLNBias, info.TimeMixLNBias},
				{&result.TimeMixOutput, info.TimeMixOutput},
				{&result.ChannelMixLerpK, info.ChannelMixLerpK},
				{&result.ChannelMixLerpR, info.ChannelMixLerpR},
				{&result.ChannelMixKey, info.ChannelMixKey},
				{&result.ChannelMixValue, info.ChannelMixValue},
				{&result.ChannelMixReceptance, info.ChannelMixReceptance},
			} {
				optionalItems = append(optionalItems, struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{item.destination, item.info})
			}
		} else if kimi {
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
			optionalItems = append(optionalItems,
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMQueryConv, info.SSMQueryConv},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMKeyConv, info.SSMKeyConv},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMValueConv, info.SSMValueConv},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMForgetA, info.SSMForgetA},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMForgetB, info.SSMForgetB},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMBeta, info.SSMBeta},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMA, info.SSMA},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMTimeStep, info.SSMTimeStep},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMOutputGateA, info.SSMOutputGateA},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMOutputGateB, info.SSMOutputGateB},
				struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMNorm, info.SSMNorm},
			)
		} else if info.ShortConvKernel != nil {
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
			if info.SSMX != nil {
				optionalItems = append(optionalItems,
					struct {
						destination **reference.Value
						info        *gguf.TensorInfo
					}{&result.SSMX, info.SSMX},
					struct {
						destination **reference.Value
						info        *gguf.TensorInfo
					}{&result.SSMTimeStepWeight, info.SSMTimeStepWeight},
				)
				if info.SSMTimeStepNorm != nil {
					optionalItems = append(optionalItems,
						struct {
							destination **reference.Value
							info        *gguf.TensorInfo
						}{&result.SSMTimeStepNorm, info.SSMTimeStepNorm},
						struct {
							destination **reference.Value
							info        *gguf.TensorInfo
						}{&result.SSMBNorm, info.SSMBNorm},
						struct {
							destination **reference.Value
							info        *gguf.TensorInfo
						}{&result.SSMCNorm, info.SSMCNorm},
					)
				}
			} else {
				optionalItems = append(optionalItems, struct {
					destination **reference.Value
					info        *gguf.TensorInfo
				}{&result.SSMNorm, info.SSMNorm})
			}
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
		{info.FeedForwardActivationScale, &result.FeedForwardActivationScale},
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
		{info.FeedForwardLatentDown, &result.FeedForwardLatentDown},
		{info.FeedForwardLatentUp, &result.FeedForwardLatentUp},
		{info.FeedForwardSharedGate, &result.FeedForwardSharedGate},
		{info.FeedForwardSharedUp, &result.FeedForwardSharedUp},
		{info.FeedForwardSharedDown, &result.FeedForwardSharedDown},
		{info.FeedForwardSharedRouter, &result.FeedForwardSharedRouter},
		{info.LayerOutputScale, &result.LayerOutputScale},
		{info.PerLayerInputGate, &result.PerLayerInputGate},
		{info.PerLayerProjection, &result.PerLayerProjection},
		{info.PerLayerPostNorm, &result.PerLayerPostNorm},
		{info.AltUpCorrectCoefficient, &result.AltUpCorrectCoefficient},
		{info.AltUpCorrectScale, &result.AltUpCorrectScale},
		{info.AltUpPredictCoefficient, &result.AltUpPredictCoefficient},
		{info.AltUpRouter, &result.AltUpRouter},
		{info.AltUpRouterNorm, &result.AltUpRouterNorm},
		{info.LaurelLeft, &result.LaurelLeft},
		{info.LaurelRight, &result.LaurelRight},
		{info.LaurelPostNorm, &result.LaurelPostNorm},
		{info.AttentionKVAMQA, &result.AttentionKVAMQA},
		{info.AttentionKVANorm, &result.AttentionKVANorm},
		{info.AttentionKVB, &result.AttentionKVB},
		{info.AttentionKB, &result.AttentionKB},
		{info.AttentionVB, &result.AttentionVB},
		{info.IndexerKNorm, &result.IndexerKNorm},
		{info.IndexerKNormBias, &result.IndexerKNormBias},
		{info.IndexerProjection, &result.IndexerProjection},
		{info.IndexerAttentionK, &result.IndexerAttentionK},
		{info.IndexerAttentionQB, &result.IndexerAttentionQB},
		{info.AttentionOutputA, &result.AttentionOutputA},
		{info.AttentionCompressorKV, &result.AttentionCompressorKV},
		{info.AttentionCompressorGate, &result.AttentionCompressorGate},
		{info.AttentionCompressorAPE, &result.AttentionCompressorAPE},
		{info.AttentionCompressorNorm, &result.AttentionCompressorNorm},
		{info.IndexerCompressorKV, &result.IndexerCompressorKV},
		{info.IndexerCompressorGate, &result.IndexerCompressorGate},
		{info.IndexerCompressorAPE, &result.IndexerCompressorAPE},
		{info.IndexerCompressorNorm, &result.IndexerCompressorNorm},
		{info.HyperAttentionFN, &result.HyperAttentionFN},
		{info.HyperAttentionBase, &result.HyperAttentionBase},
		{info.HyperAttentionScale, &result.HyperAttentionScale},
		{info.HyperFeedForwardFN, &result.HyperFeedForwardFN},
		{info.HyperFeedForwardBase, &result.HyperFeedForwardBase},
		{info.HyperFeedForwardScale, &result.HyperFeedForwardScale},
		{info.HyperHeadFN, &result.HyperHeadFN},
		{info.HyperHeadBase, &result.HyperHeadBase},
		{info.HyperHeadScale, &result.HyperHeadScale},
		{info.FeedForwardHashExperts, &result.FeedForwardHashExperts},
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
	for _, item := range []struct {
		info        *gguf.TensorInfo
		destination **reference.Value
	}{
		{info.CrossAttentionNorm, &result.CrossAttentionNorm},
		{info.CrossAttentionQ, &result.CrossAttentionQ},
		{info.CrossAttentionK, &result.CrossAttentionK},
		{info.CrossAttentionV, &result.CrossAttentionV},
		{info.CrossAttentionOutput, &result.CrossAttentionOutput},
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
	if layer.TimeMixW1 != nil {
		for _, item := range []struct {
			name        string
			value       *reference.Value
			destination **tensor.Tensor
		}{
			{"time_mix_w1.weight", layer.TimeMixW1, &result.TimeMixW1},
			{"time_mix_w2.weight", layer.TimeMixW2, &result.TimeMixW2},
			{"time_mix_w0.weight", layer.TimeMixW0, &result.TimeMixW0},
			{"time_mix_a0.weight", layer.TimeMixA0, &result.TimeMixA0},
			{"time_mix_a1.weight", layer.TimeMixA1, &result.TimeMixA1},
			{"time_mix_a2.weight", layer.TimeMixA2, &result.TimeMixA2},
			{"time_mix_v0.weight", layer.TimeMixV0, &result.TimeMixV0},
			{"time_mix_v1.weight", layer.TimeMixV1, &result.TimeMixV1},
			{"time_mix_v2.weight", layer.TimeMixV2, &result.TimeMixV2},
			{"time_mix_g1.weight", layer.TimeMixG1, &result.TimeMixG1},
			{"time_mix_g2.weight", layer.TimeMixG2, &result.TimeMixG2},
			{"time_mix_k_k.weight", layer.TimeMixKK, &result.TimeMixKK},
			{"time_mix_k_a.weight", layer.TimeMixKA, &result.TimeMixKA},
			{"time_mix_r_k.weight", layer.TimeMixRK, &result.TimeMixRK},
			{"time_mix_lerp_x.weight", layer.TimeMixLerpX, &result.TimeMixLerpX},
			{"time_mix_lerp_fused.weight", layer.TimeMixLerpFused, &result.TimeMixLerpFused},
			{"time_mix_lerp_w.weight", layer.TimeMixLerpW, &result.TimeMixLerpW},
			{"time_mix_lerp_k.weight", layer.TimeMixLerpK, &result.TimeMixLerpK},
			{"time_mix_lerp_v.weight", layer.TimeMixLerpV, &result.TimeMixLerpV},
			{"time_mix_lerp_r.weight", layer.TimeMixLerpR, &result.TimeMixLerpR},
			{"time_mix_lerp_g.weight", layer.TimeMixLerpG, &result.TimeMixLerpG},
			{"time_mix_first.weight", layer.TimeMixFirst, &result.TimeMixFirst},
			{"time_mix_decay.weight", layer.TimeMixDecay, &result.TimeMixDecay},
			{"time_mix_decay_w1.weight", layer.TimeMixDecayW1, &result.TimeMixDecayW1},
			{"time_mix_decay_w2.weight", layer.TimeMixDecayW2, &result.TimeMixDecayW2},
			{"time_mix_key.weight", layer.TimeMixKey, &result.TimeMixKey},
			{"time_mix_value.weight", layer.TimeMixValue, &result.TimeMixValue},
			{"time_mix_receptance.weight", layer.TimeMixReceptance, &result.TimeMixReceptance},
			{"time_mix_gate.weight", layer.TimeMixGate, &result.TimeMixGate},
			{"time_mix_ln.weight", layer.TimeMixLN, &result.TimeMixLN},
			{"time_mix_ln.bias", layer.TimeMixLNBias, &result.TimeMixLNBias},
			{"time_mix_output.weight", layer.TimeMixOutput, &result.TimeMixOutput},
			{"channel_mix_lerp_k.weight", layer.ChannelMixLerpK, &result.ChannelMixLerpK},
			{"channel_mix_lerp_r.weight", layer.ChannelMixLerpR, &result.ChannelMixLerpR},
			{"channel_mix_key.weight", layer.ChannelMixKey, &result.ChannelMixKey},
			{"channel_mix_value.weight", layer.ChannelMixValue, &result.ChannelMixValue},
			{"channel_mix_receptance.weight", layer.ChannelMixReceptance, &result.ChannelMixReceptance},
		} {
			if item.value == nil {
				continue
			}
			*item.destination = input(item.name, *item.value)
		}
	} else if layer.SSMInput != nil && layer.AttentionOutput.Shape.Rank == 0 {
		result.SSMInput = input("ssm_in.weight", *layer.SSMInput)
		result.SSMConv1D = input("ssm_conv1d.weight", *layer.SSMConv1D)
		if layer.SSMConv1DBias != nil {
			result.SSMConv1DBias = input("ssm_conv1d.bias", *layer.SSMConv1DBias)
		}
		result.SSMTimeStep = input("ssm_dt.bias", *layer.SSMTimeStep)
		result.SSMA = input("ssm_a", *layer.SSMA)
		result.SSMD = input("ssm_d", *layer.SSMD)
		result.SSMOutput = input("ssm_out.weight", *layer.SSMOutput)
		if layer.SSMX != nil {
			result.SSMX = input("ssm_x.weight", *layer.SSMX)
			result.SSMTimeStepWeight = input("ssm_dt.weight", *layer.SSMTimeStepWeight)
			if layer.SSMTimeStepNorm != nil {
				result.SSMTimeStepNorm = input("ssm_dt_norm.weight", *layer.SSMTimeStepNorm)
				result.SSMBNorm = input("ssm_b_norm.weight", *layer.SSMBNorm)
				result.SSMCNorm = input("ssm_c_norm.weight", *layer.SSMCNorm)
			}
		} else if layer.SSMNorm != nil {
			result.SSMNorm = input("ssm_norm.weight", *layer.SSMNorm)
		}
	} else if layer.SSMQueryConv != nil {
		result.AttentionQ = input("attn_q.weight", layer.AttentionQ)
		result.AttentionK = input("attn_k.weight", layer.AttentionK)
		result.AttentionV = input("attn_v.weight", layer.AttentionV)
		result.AttentionOutput = input("attn_output.weight", layer.AttentionOutput)
		for _, item := range []struct {
			name        string
			value       *reference.Value
			destination **tensor.Tensor
		}{
			{"ssm_conv1d_q.weight", layer.SSMQueryConv, &result.SSMQueryConv},
			{"ssm_conv1d_k.weight", layer.SSMKeyConv, &result.SSMKeyConv},
			{"ssm_conv1d_v.weight", layer.SSMValueConv, &result.SSMValueConv},
			{"ssm_f_a.weight", layer.SSMForgetA, &result.SSMForgetA},
			{"ssm_f_b.weight", layer.SSMForgetB, &result.SSMForgetB},
			{"ssm_beta.weight", layer.SSMBeta, &result.SSMBeta},
			{"ssm_a", layer.SSMA, &result.SSMA},
			{"ssm_dt.bias", layer.SSMTimeStep, &result.SSMTimeStep},
			{"ssm_g_a.weight", layer.SSMOutputGateA, &result.SSMOutputGateA},
			{"ssm_g_b.weight", layer.SSMOutputGateB, &result.SSMOutputGateB},
			{"ssm_norm.weight", layer.SSMNorm, &result.SSMNorm},
		} {
			*item.destination = input(item.name, *item.value)
		}
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
	if layer.SSMInput != nil && layer.AttentionOutput.Shape.Rank != 0 {
		result.SSMInput = input("ssm_in.weight", *layer.SSMInput)
		result.SSMConv1D = input("ssm_conv1d.weight", *layer.SSMConv1D)
		if layer.SSMConv1DBias != nil {
			result.SSMConv1DBias = input("ssm_conv1d.bias", *layer.SSMConv1DBias)
		}
		result.SSMTimeStep = input("ssm_dt.bias", *layer.SSMTimeStep)
		result.SSMA = input("ssm_a", *layer.SSMA)
		result.SSMD = input("ssm_d", *layer.SSMD)
		if layer.SSMNorm != nil {
			result.SSMNorm = input("ssm_norm.weight", *layer.SSMNorm)
		}
		result.SSMOutput = input("ssm_out.weight", *layer.SSMOutput)
	}
	bindHostLayerGraphFields(builder, prefix, layer, &result, feeds)
	if err := builder.Err(); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	return result, feeds, nil
}
