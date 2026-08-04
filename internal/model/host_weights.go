package model

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/quant"
	"llamacpp2go/internal/tensor"
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

// LoadHostLayer materializes one releasable layer.
func LoadHostLayer(
	ctx context.Context,
	file *gguf.File,
	info LayerWeights,
) (HostLayer, error) {
	var result HostLayer
	if !info.Recurrent && info.SSMInput != nil &&
		(info.SSMConv1D == nil || info.SSMTimeStep == nil || info.SSMA == nil ||
			info.SSMD == nil || info.SSMOutput == nil) {
		return HostLayer{}, errors.New("host hybrid SSM catalog is incomplete")
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
	}
	if err := loadHostLayerGraphFields(ctx, file, &info, &result); err != nil {
		return HostLayer{}, err
	}
	return result, nil
}

// GraphInputs builds zero-copy graph feeds.
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
	result := LayerGraphWeights{}
	bindHostLayerGraphFields(builder, prefix, layer, &result, feeds)
	if err := builder.Err(); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	return result, feeds, nil
}
