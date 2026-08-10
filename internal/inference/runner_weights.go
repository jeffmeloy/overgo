package inference

import (
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/model"
)

func f32RequiredModelTensors(weights model.Weights) map[string]struct{} {
	result := make(map[string]struct{})
	layers := slices.Clone(weights.Layers)
	for _, draft := range weights.DraftCatalogs() {
		layers = append(layers, draft.Layer)
	}
	for _, layer := range layers {
		for _, info := range []*gguf.TensorInfo{
			layer.SSMConv1D,
			layer.SSMQueryConv,
			layer.SSMKeyConv,
			layer.SSMValueConv,
			layer.ShortConvKernel,
		} {
			if info != nil {
				result[info.Name] = struct{}{}
			}
		}
	}
	return result
}

// getRowsSourceTensors: tensors consumed by get_rows (embedding tables), which
// have no native-F16 kernel and must stay F32-resident.
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
