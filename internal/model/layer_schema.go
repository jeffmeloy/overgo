package model

import (
	"context"
	"reflect"
	"strings"
	"unicode"

	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// layerTensorSchema is the single declaration of every dense-layer tensor
// slot. The inventory (gguf.TensorInfo), host (reference.Value), and graph
// (tensor.Tensor) views are defined types over one instantiation each, so
// the three views share field names and indexes by construction and cannot
// diverge. A nil slot means the tensor is absent for the architecture.
type layerTensorSchema[T any] struct {
	Recurrent                   bool
	AttentionNorm               *T
	AttentionNormBias           *T
	AttentionNorm2              *T
	AttentionNorm2Bias          *T
	AttentionQ                  *T
	AttentionQB                 *T
	AttentionK                  *T
	AttentionV                  *T
	AttentionOutput             *T
	AttentionQScale             *T
	AttentionKScale             *T
	AttentionVScale             *T
	AttentionOutputScale        *T
	AttentionTemperatureScale   *T
	AttentionSubNorm            *T
	AttentionQBias              *T
	AttentionKBias              *T
	AttentionVBias              *T
	AttentionOutputBias         *T
	AttentionQNorm              *T
	AttentionKNorm              *T
	AttentionQNormBias          *T
	AttentionKNormBias          *T
	AttentionPostNorm           *T
	AttentionPostNormBias       *T
	AttentionRelativeBias       *T
	CrossAttentionNorm          *T
	CrossAttentionQ             *T
	CrossAttentionK             *T
	CrossAttentionV             *T
	CrossAttentionOutput        *T
	AttentionOutputGate         *T
	AttentionSinks              *T
	AttentionBlockIDs           *T
	RopeFactors                 *T
	FeedForwardNorm             *T
	FeedForwardNormBias         *T
	FeedForwardExpertNorm       *T
	FeedForwardGate             *T
	FeedForwardUp               *T
	FeedForwardDown             *T
	FeedForwardGateScale        *T
	FeedForwardUpScale          *T
	FeedForwardDownScale        *T
	FeedForwardActivationScale  *T
	FeedForwardSubNorm          *T
	FeedForwardGateBias         *T
	FeedForwardUpBias           *T
	FeedForwardDownBias         *T
	FeedForwardPostNorm         *T
	FeedForwardPostNormBias     *T
	FeedForwardPreNorm2         *T
	FeedForwardPostNorm1        *T
	FeedForwardPostNorm2        *T
	FeedForwardRouter           *T
	FeedForwardRouterBias       *T
	FeedForwardRouterScale      *T
	FeedForwardGateUpExperts    *T
	FeedForwardGateExperts      *T
	FeedForwardUpExperts        *T
	FeedForwardDownExperts      *T
	FeedForwardDownExpertsScale *T
	FeedForwardGateChunkExperts *T
	FeedForwardUpChunkExperts   *T
	FeedForwardDownChunkExperts *T
	FeedForwardExpertBias       *T
	FeedForwardLatentDown       *T
	FeedForwardLatentUp         *T
	FeedForwardSharedGate       *T
	FeedForwardSharedUp         *T
	FeedForwardSharedDown       *T
	FeedForwardSharedRouter     *T
	LayerOutputScale            *T
	EmbeddingSkip               *T
	PerLayerInput               *T
	PerLayerInputGate           *T
	PerLayerProjection          *T
	PerLayerPostNorm            *T
	ShortConvKernel             *T
	ShortConvInput              *T
	ShortConvOutput             *T
	AttentionKVAMQA             *T
	AttentionKVANorm            *T
	AttentionKVB                *T
	AttentionKB                 *T
	AttentionVB                 *T
	IndexerKNorm                *T
	IndexerKNormBias            *T
	IndexerProjection           *T
	IndexerAttentionK           *T
	IndexerAttentionQB          *T
	AttentionOutputA            *T
	AttentionCompressorKV       *T
	AttentionCompressorGate     *T
	AttentionCompressorAPE      *T
	AttentionCompressorNorm     *T
	IndexerCompressorKV         *T
	IndexerCompressorGate       *T
	IndexerCompressorAPE        *T
	IndexerCompressorNorm       *T
	HyperAttentionFN            *T
	HyperAttentionBase          *T
	HyperAttentionScale         *T
	HyperFeedForwardFN          *T
	HyperFeedForwardBase        *T
	HyperFeedForwardScale       *T
	HyperHeadFN                 *T
	HyperHeadBase               *T
	HyperHeadScale              *T
	FeedForwardHashExperts      *T
	VisualAttentionQKV          *T
	VisualAttentionOutput       *T
	VisualFeedForwardGate       *T
	VisualFeedForwardUp         *T
	VisualFeedForwardDown       *T
	AltUpCorrectCoefficient     *T
	AltUpCorrectScale           *T
	AltUpPredictCoefficient     *T
	AltUpRouter                 *T
	AltUpRouterNorm             *T
	LaurelLeft                  *T
	LaurelRight                 *T
	LaurelPostNorm              *T

	AttentionQKV         *T
	AttentionQKVBias     *T
	AttentionGate        *T
	SSMConv1D            *T
	SSMConv1DBias        *T
	SSMInput             *T
	SSMX                 *T
	SSMTimeStepWeight    *T
	SSMTimeStep          *T
	SSMTimeStepNorm      *T
	SSMA                 *T
	SSMD                 *T
	SSMBNorm             *T
	SSMCNorm             *T
	SSMBeta              *T
	SSMAlpha             *T
	SSMBetaAlpha         *T
	SSMNorm              *T
	SSMOutput            *T
	SSMQueryConv         *T
	SSMKeyConv           *T
	SSMValueConv         *T
	SSMForgetA           *T
	SSMForgetB           *T
	SSMOutputGateA       *T
	SSMOutputGateB       *T
	TimeMixW1            *T
	TimeMixW2            *T
	TimeMixW0            *T
	TimeMixA0            *T
	TimeMixA1            *T
	TimeMixA2            *T
	TimeMixV0            *T
	TimeMixV1            *T
	TimeMixV2            *T
	TimeMixG1            *T
	TimeMixG2            *T
	TimeMixKK            *T
	TimeMixKA            *T
	TimeMixRK            *T
	TimeMixLerpX         *T
	TimeMixLerpFused     *T
	TimeMixLerpW         *T
	TimeMixLerpK         *T
	TimeMixLerpV         *T
	TimeMixLerpR         *T
	TimeMixLerpG         *T
	TimeMixFirst         *T
	TimeMixDecay         *T
	TimeMixDecayW1       *T
	TimeMixDecayW2       *T
	TimeMixKey           *T
	TimeMixValue         *T
	TimeMixReceptance    *T
	TimeMixGate          *T
	TimeMixLN            *T
	TimeMixLNBias        *T
	TimeMixOutput        *T
	ChannelMixLerpK      *T
	ChannelMixLerpR      *T
	ChannelMixKey        *T
	ChannelMixValue      *T
	ChannelMixReceptance *T
}

// LayerWeights: dense-layer tensor inventory (validated GGUF metadata).
type LayerWeights = layerTensorSchema[gguf.TensorInfo]

// HostLayer: one dense layer dequantized to contiguous F32 values.
type HostLayer layerTensorSchema[reference.Value]

// LayerGraphWeights: graph inputs for one dense decoder block.
type LayerGraphWeights = layerTensorSchema[tensor.Tensor]

// layerBindingSlot: one compiled catalog slot shared by all schema views.
type layerBindingSlot struct {
	name      string
	inputName string
	index     int
}

// layerRuntimeOnlySlots never enter the weight-catalog binding: they are
// either graph nodes synthesized at runtime or tensors bound by a
// dedicated projection pipeline.
var layerRuntimeOnlySlots = map[string]bool{
	"AttentionTemperatureScale": true,
	"AttentionBlockIDs":         true,
	"EmbeddingSkip":             true,
	"PerLayerInput":             true,
	"VisualAttentionQKV":        true,
	"VisualAttentionOutput":     true,
	"VisualFeedForwardGate":     true,
	"VisualFeedForwardUp":       true,
	"VisualFeedForwardDown":     true,
}

// layerBindingSchema is compiled once from the unified layer schema.
var layerBindingSchema = compileLayerBindingSchema()

func compileLayerBindingSchema() []layerBindingSlot {
	schema := reflect.TypeOf(layerTensorSchema[gguf.TensorInfo]{})
	slots := make([]layerBindingSlot, 0, schema.NumField())
	for index := range schema.NumField() {
		field := schema.Field(index)
		if field.Type.Kind() != reflect.Pointer || layerRuntimeOnlySlots[field.Name] {
			continue
		}
		slots = append(slots, layerBindingSlot{
			name: field.Name, inputName: graphInputName(field.Name), index: index,
		})
	}
	return slots
}

func graphInputName(name string) string {
	items := []rune(name)
	var output strings.Builder
	for index, item := range items {
		if unicode.IsUpper(item) && index > 0 &&
			(unicode.IsLower(items[index-1]) ||
				(index+1 < len(items) && unicode.IsLower(items[index+1]))) {
			output.WriteByte('_')
		}
		output.WriteRune(unicode.ToLower(item))
	}
	return output.String()
}

type hostTensorLoader func(context.Context, *gguf.File, gguf.TensorInfo) (reference.Value, error)

func loadHostLayerGraphFields(
	ctx context.Context,
	file *gguf.File,
	info *LayerWeights,
	result *HostLayer,
) error {
	return loadHostLayerGraphFieldsWith(ctx, file, info, result, LoadHostTensor)
}

func loadHostLayerGraphFieldsWith(
	ctx context.Context,
	file *gguf.File,
	info *LayerWeights,
	result *HostLayer,
	load hostTensorLoader,
) error {
	infoValue := reflect.ValueOf(info).Elem()
	hostValue := reflect.ValueOf(result).Elem()
	for _, slot := range layerBindingSchema {
		tensorInfo := infoValue.Field(slot.index).Interface().(*gguf.TensorInfo)
		if tensorInfo == nil {
			continue
		}
		value, err := load(ctx, file, *tensorInfo)
		if err != nil {
			return err
		}
		hostValue.Field(slot.index).Set(reflect.ValueOf(&value))
	}
	return nil
}

func bindLayerGraphFields[T any](
	source *layerTensorSchema[T],
	result *LayerGraphWeights,
	bind func(layerBindingSlot, *T) (*tensor.Tensor, error),
) error {
	sourceValue := reflect.ValueOf(source).Elem()
	graphValue := reflect.ValueOf(result).Elem()
	for _, slot := range layerBindingSchema {
		value := sourceValue.Field(slot.index).Interface().(*T)
		if value == nil {
			continue
		}
		node, err := bind(slot, value)
		if err != nil {
			return err
		}
		graphValue.Field(slot.index).Set(reflect.ValueOf(node))
	}
	return nil
}

// DeviceTensorBinder: metadata-to-device graph binding.
type DeviceTensorBinder func(
	*tensor.Builder,
	gguf.TensorInfo,
) (*tensor.Tensor, driver.DevicePtr, error)

func bindDeviceLayerGraphFields(
	bind DeviceTensorBinder,
	builder *tensor.Builder,
	info *LayerWeights,
	result *LayerGraphWeights,
	feeds *tensor.InputBindings[driver.DevicePtr],
) error {
	return bindLayerGraphFields(info, result, func(_ layerBindingSlot, tensorInfo *gguf.TensorInfo) (*tensor.Tensor, error) {
		node, pointer, err := bind(builder, *tensorInfo)
		if err != nil {
			return nil, err
		}
		feeds.Add(node, pointer)
		return node, nil
	})
}
