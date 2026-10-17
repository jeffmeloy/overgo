package inference

import (
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/dtype"
)

// catalogLayers: the model's layers, then each draft head's.
func catalogLayers(weights model.Weights) []model.LayerWeights {
	layers := slices.Clone(weights.Layers)
	for _, draft := range weights.DraftCatalogs() {
		layers = append(layers, draft.Layer)
	}
	return layers
}

// expertBankRank: a routed-expert bank is [inner, rows, experts].
const expertBankRank = 3

// nativeExpertBankTensors: the BF16 routed-expert banks that stay at file
// width on the device, where the routed-expert kernels widen each element as
// they read it. A layer's banks qualify together or not at all: the kernels
// take one storage type per node, and a LoRA merge needs an F32 base.
func nativeExpertBankTensors(weights model.Weights, adapted map[string]struct{}) map[string]struct{} {
	result := make(map[string]struct{})
	for _, layer := range catalogLayers(weights) {
		banks := slices.DeleteFunc([]*gguf.TensorInfo{
			layer.FeedForwardGateUpExperts,
			layer.FeedForwardGateExperts,
			layer.FeedForwardUpExperts,
			layer.FeedForwardDownExperts,
		}, func(info *gguf.TensorInfo) bool { return info == nil })
		if len(banks) == 0 || slices.ContainsFunc(banks, func(info *gguf.TensorInfo) bool {
			_, isAdapted := adapted[info.Name]
			return isAdapted || info.Type != dtype.BF16 || info.Dimensions != expertBankRank
		}) {
			continue
		}
		for _, info := range banks {
			result[info.Name] = struct{}{}
		}
	}
	return result
}

func f32RequiredModelTensors(weights model.Weights) map[string]struct{} {
	result := make(map[string]struct{})
	for _, layer := range catalogLayers(weights) {
		for _, info := range []*gguf.TensorInfo{
			layer.SSMConv1D,
			layer.SSMQueryConv,
			layer.SSMKeyConv,
			layer.SSMValueConv,
			layer.ShortConvKernel,
			layer.NGramConv,
		} {
			if info != nil {
				result[info.Name] = struct{}{}
			}
		}
	}
	return result
}

// getRowsSourceTensors: embedding table inventory.
func getRowsSourceTensors(weights model.Weights) map[string]struct{} {
	result := make(map[string]struct{})
	result[weights.TokenEmbedding.Name] = struct{}{}
	for _, info := range []*gguf.TensorInfo{
		weights.PositionEmbedding,
		weights.PerLayerTokenEmbedding,
	} {
		if info != nil {
			result[info.Name] = struct{}{}
		}
	}
	return result
}

func selectedModelTensors(file *gguf.File, weights model.Weights) []gguf.TensorInfo {
	names := make(map[string]struct{})
	for _, info := range weights.TensorInfos() {
		if weights.NGramTokenEmbedding != nil && info.Name == weights.NGramTokenEmbedding.Name {
			continue
		}
		names[info.Name] = struct{}{}
	}
	result := make([]gguf.TensorInfo, 0, len(names))
	for _, info := range file.Tensors {
		if _, ok := names[info.Name]; ok {
			result = append(result, info)
		}
	}
	return result
}
