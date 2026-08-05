package hfgguf

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"strconv"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/hfrepo"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/safetensors"
	"llamacpp2go/internal/strictjson"
)

var denseLayerNames = map[string]string{
	"input_layernorm.weight":          "attn_norm.weight",
	"post_attention_layernorm.weight": "ffn_norm.weight",
	"self_attn.q_proj.weight":         "attn_q.weight",
	"self_attn.q_proj.bias":           "attn_q.bias",
	"self_attn.k_proj.weight":         "attn_k.weight",
	"self_attn.k_proj.bias":           "attn_k.bias",
	"self_attn.v_proj.weight":         "attn_v.weight",
	"self_attn.v_proj.bias":           "attn_v.bias",
	"self_attn.o_proj.weight":         "attn_output.weight",
	"self_attn.o_proj.bias":           "attn_output.bias",
	"mlp.gate_proj.weight":            "ffn_gate.weight",
	"mlp.gate_proj.bias":              "ffn_gate.bias",
	"mlp.up_proj.weight":              "ffn_up.weight",
	"mlp.up_proj.bias":                "ffn_up.bias",
	"mlp.down_proj.weight":            "ffn_down.weight",
	"mlp.down_proj.bias":              "ffn_down.bias",
}

// ValidateDenseRepository: adapt standard Llama/Qwen2 metadata and catalogs.
func ValidateDenseRepository(repository *hfrepo.Repository) (model.Spec, error) {
	if repository == nil || repository.Tensors == nil {
		return model.Spec{}, errors.New("HF/GGUF adapter: nil repository")
	}
	architecture := repository.Identity.ModelType
	if architecture != "llama" && architecture != "qwen2" {
		return model.Spec{}, fmt.Errorf("HF/GGUF adapter: unsupported model type %q", architecture)
	}
	metadata, err := denseMetadata(repository, architecture)
	if err != nil {
		return model.Spec{}, err
	}
	tensors, err := denseTensorCatalog(repository.Tensors)
	if err != nil {
		return model.Spec{}, err
	}
	file := &gguf.File{Metadata: metadata, Tensors: tensors}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return model.Spec{}, fmt.Errorf("HF/GGUF adapter: model spec: %w", err)
	}
	if _, err := model.ReadWeights(file, spec); err != nil {
		return model.Spec{}, fmt.Errorf("HF/GGUF adapter: weight catalog: %w", err)
	}
	return spec, nil
}

func denseMetadata(repository *hfrepo.Repository, architecture string) ([]gguf.Metadata, error) {
	contextLength, err := required[uint32](repository.Config, "max_position_embeddings")
	if err != nil {
		return nil, err
	}
	embeddingLength, err := required[uint32](repository.Config, "hidden_size")
	if err != nil {
		return nil, err
	}
	blockCount, err := required[uint32](repository.Config, "num_hidden_layers")
	if err != nil {
		return nil, err
	}
	feedForwardLength, err := required[uint32](repository.Config, "intermediate_size")
	if err != nil {
		return nil, err
	}
	headCount, err := required[uint32](repository.Config, "num_attention_heads")
	if err != nil {
		return nil, err
	}
	kvHeadCount, err := required[uint32](repository.Config, "num_key_value_heads")
	if err != nil {
		return nil, err
	}
	if headCount == 0 || embeddingLength == 0 {
		return nil, errors.New("HF/GGUF adapter: invalid attention dimensions")
	}
	headLength, ok, err := optional[uint32](repository.Config, "head_dim")
	if err != nil {
		return nil, err
	}
	if !ok {
		if embeddingLength%headCount != 0 {
			return nil, errors.New("HF/GGUF adapter: hidden size is not divisible by head count")
		}
		headLength = embeddingLength / headCount
	}
	ropeBase, err := required[float32](repository.Config, "rope_theta")
	if err != nil {
		return nil, err
	}
	normEpsilon, err := required[float32](repository.Config, "rms_norm_eps")
	if err != nil {
		return nil, err
	}
	vocabulary, err := required[uint32](repository.Config, "vocab_size")
	if err != nil {
		return nil, err
	}
	prefix := architecture + "."
	return []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, architecture),
		metadata("general.name", gguf.ValueTypeString, filepath.Base(repository.Directory)),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, blockCount),
		metadata(prefix+"context_length", gguf.ValueTypeUint32, contextLength),
		metadata(prefix+"embedding_length", gguf.ValueTypeUint32, embeddingLength),
		metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, feedForwardLength),
		metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, headCount),
		metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, kvHeadCount),
		metadata(prefix+"attention.key_length", gguf.ValueTypeUint32, headLength),
		metadata(prefix+"attention.value_length", gguf.ValueTypeUint32, headLength),
		metadata(prefix+"rope.freq_base", gguf.ValueTypeFloat32, ropeBase),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, normEpsilon),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, vocabulary),
	}, nil
}

func denseTensorCatalog(source *safetensors.Source) ([]gguf.TensorInfo, error) {
	mappings, err := denseTensorMappings(source)
	if err != nil {
		return nil, err
	}
	return mappedTensorCatalog(mappings)
}

type tensorMapping struct {
	name   string
	tensor safetensors.Tensor
	shape  []uint64
}

func mappedTensorCatalog(mappings []tensorMapping) ([]gguf.TensorInfo, error) {
	tensors := make([]gguf.TensorInfo, 0, len(mappings))
	for _, mapping := range mappings {
		tensor, name := mapping.tensor, mapping.name
		shape := mapping.sourceShape()
		storage, ok := ggufStorage(tensor.DType, len(shape) == 1)
		if !ok {
			return nil, fmt.Errorf("HF/GGUF adapter: tensor %q dtype %q needs conversion", tensor.Name, tensor.DType)
		}
		info := gguf.TensorInfo{Name: name, Type: storage, Dimensions: uint32(len(shape))}
		for index, dimension := range shape {
			info.Shape[len(shape)-1-index] = dimension
		}
		traits, _ := storage.Traits()
		elements := tensor.Elements()
		if elements > math.MaxUint64/traits.TypeSize {
			return nil, fmt.Errorf("HF/GGUF adapter: tensor %q size overflows", tensor.Name)
		}
		info.Size = elements * traits.TypeSize
		tensors = append(tensors, info)
	}
	return tensors, nil
}

func (m tensorMapping) sourceShape() []uint64 {
	if m.shape != nil {
		return m.shape
	}
	return m.tensor.Shape
}

func denseTensorMappings(source *safetensors.Source) ([]tensorMapping, error) {
	mappings := make([]tensorMapping, 0, len(source.Tensors))
	seen := make(map[string]string, len(source.Tensors))
	for _, sourceName := range source.Names() {
		tensor := source.Tensors[sourceName]
		name, include, err := denseTensorName(sourceName)
		if err != nil {
			return nil, err
		}
		if !include {
			continue
		}
		if previous, duplicate := seen[name]; duplicate {
			return nil, fmt.Errorf("HF/GGUF adapter: tensors %q and %q both map to %q", previous, sourceName, name)
		}
		seen[name] = sourceName
		if len(tensor.Shape) == 0 || len(tensor.Shape) > gguf.MaxDimensions {
			return nil, fmt.Errorf("HF/GGUF adapter: tensor %q rank %d is unsupported", sourceName, len(tensor.Shape))
		}
		mappings = append(mappings, tensorMapping{name: name, tensor: tensor})
	}
	return mappings, nil
}

// DenseTensorData: stream standard dense payloads in GGUF tensor order.
func DenseTensorData(repository *hfrepo.Repository) ([]gguf.TensorData, error) {
	if repository == nil || repository.Tensors == nil {
		return nil, errors.New("HF/GGUF adapter: nil repository")
	}
	mappings, err := denseTensorMappings(repository.Tensors)
	if err != nil {
		return nil, err
	}
	return mappedTensorData(mappings)
}

func mappedTensorData(mappings []tensorMapping) ([]gguf.TensorData, error) {
	tensors := make([]gguf.TensorData, 0, len(mappings))
	for _, mapping := range mappings {
		tensor := mapping.tensor
		shape := mapping.sourceShape()
		storage, ok := ggufStorage(tensor.DType, len(shape) == 1)
		if !ok {
			return nil, fmt.Errorf("HF/GGUF adapter: tensor %q dtype %q needs conversion", tensor.Name, tensor.DType)
		}
		var reader io.Reader = tensor.Reader()
		if len(shape) == 1 && tensor.DType != "F32" {
			converted, err := safetensors.F32Reader(tensor)
			if err != nil {
				return nil, fmt.Errorf("HF/GGUF adapter: tensor %q: %w", tensor.Name, err)
			}
			reader = converted
		}
		ggmlShape := make([]uint64, len(shape))
		for index, dimension := range shape {
			ggmlShape[len(shape)-1-index] = dimension
		}
		tensors = append(tensors, gguf.TensorData{
			Name: mapping.name, Shape: ggmlShape, Type: storage, Data: reader,
		})
	}
	return tensors, nil
}

func denseTensorName(name string) (string, bool, error) {
	switch name {
	case "model.embed_tokens.weight":
		return "token_embd.weight", true, nil
	case "model.norm.weight":
		return "output_norm.weight", true, nil
	case "lm_head.weight":
		return "output.weight", true, nil
	case "lm_head.bias":
		return "output.bias", true, nil
	case "model.rotary_emb.inv_freq":
		return "", false, nil
	}
	rest, ok := strings.CutPrefix(name, "model.layers.")
	if !ok {
		return "", false, fmt.Errorf("HF/GGUF adapter: tensor %q has no dense mapping", name)
	}
	layer, suffix, ok := strings.Cut(rest, ".")
	if !ok {
		return "", false, fmt.Errorf("HF/GGUF adapter: tensor %q has no layer suffix", name)
	}
	index, err := strconv.ParseUint(layer, 10, 32)
	if err != nil {
		return "", false, fmt.Errorf("HF/GGUF adapter: tensor %q has invalid layer", name)
	}
	if suffix == "self_attn.rotary_emb.inv_freq" {
		return "", false, nil
	}
	destination, ok := denseLayerNames[suffix]
	if !ok {
		return "", false, fmt.Errorf("HF/GGUF adapter: tensor %q has no dense mapping", name)
	}
	return fmt.Sprintf("blk.%d.%s", index, destination), true, nil
}

func ggufStorage(dataType string, promoteVector bool) (gguf.DType, bool) {
	if promoteVector {
		switch dataType {
		case "BF16", "F16", "F32":
			return gguf.DTypeF32, true
		}
	}
	switch dataType {
	case "BF16":
		return gguf.DTypeBF16, true
	case "F16":
		return gguf.DTypeF16, true
	case "F32":
		return gguf.DTypeF32, true
	case "I16":
		return gguf.DTypeI16, true
	case "I32":
		return gguf.DTypeI32, true
	case "I64":
		return gguf.DTypeI64, true
	default:
		return 0, false
	}
}

func required[T any](config map[string]json.RawMessage, key string) (T, error) {
	value, ok, err := optional[T](config, key)
	if err != nil {
		return value, err
	}
	if !ok {
		return value, fmt.Errorf("HF/GGUF adapter: config field %q is missing", key)
	}
	return value, nil
}

func optional[T any](config map[string]json.RawMessage, key string) (T, bool, error) {
	var value T
	encoded, ok := config[key]
	if !ok || string(encoded) == "null" {
		return value, false, nil
	}
	if err := strictjson.DecodeBytes(encoded, &value); err != nil {
		return value, false, fmt.Errorf("HF/GGUF adapter: config field %q: %w", key, err)
	}
	return value, true, nil
}

func metadata(key string, valueType gguf.ValueType, value any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: valueType, Data: value}}
}
