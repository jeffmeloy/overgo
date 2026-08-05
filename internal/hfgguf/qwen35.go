package hfgguf

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strconv"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/hfrepo"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/safetensors"
)

var qwen35LayerNames = map[string]string{
	"input_layernorm.weight":          "attn_norm.weight",
	"post_attention_layernorm.weight": "post_attention_norm.weight",
	"self_attn.q_proj.weight":         "attn_q.weight",
	"self_attn.k_proj.weight":         "attn_k.weight",
	"self_attn.v_proj.weight":         "attn_v.weight",
	"self_attn.o_proj.weight":         "attn_output.weight",
	"self_attn.q_norm.weight":         "attn_q_norm.weight",
	"self_attn.k_norm.weight":         "attn_k_norm.weight",
	"mlp.gate_proj.weight":            "ffn_gate.weight",
	"mlp.up_proj.weight":              "ffn_up.weight",
	"mlp.down_proj.weight":            "ffn_down.weight",
	"linear_attn.in_proj_qkv.weight":  "attn_qkv.weight",
	"linear_attn.in_proj_z.weight":    "attn_gate.weight",
	"linear_attn.in_proj_a.weight":    "ssm_alpha.weight",
	"linear_attn.in_proj_b.weight":    "ssm_beta.weight",
	"linear_attn.A_log":               "ssm_a",
	"linear_attn.dt_bias":             "ssm_dt.bias",
	"linear_attn.conv1d.weight":       "ssm_conv1d.weight",
	"linear_attn.norm.weight":         "ssm_norm.weight",
	"linear_attn.out_proj.weight":     "ssm_out.weight",
}

// ValidateRepository: dispatch supported HF repository adapters.
func ValidateRepository(repository *hfrepo.Repository) (model.Spec, error) {
	if repository == nil {
		return model.Spec{}, errors.New("HF/GGUF adapter: nil repository")
	}
	switch repository.Identity.ModelType {
	case "llama", "qwen2":
		return ValidateDenseRepository(repository)
	case "qwen3_5":
		return ValidateQwen35Repository(repository)
	default:
		return model.Spec{}, fmt.Errorf("HF/GGUF adapter: unsupported model type %q", repository.Identity.ModelType)
	}
}

// ValidateQwen35Repository: adapt Qwen 3.5 text, recurrent, and MTP catalogs.
func ValidateQwen35Repository(repository *hfrepo.Repository) (model.Spec, error) {
	if repository == nil || repository.Tensors == nil {
		return model.Spec{}, errors.New("HF/GGUF adapter: nil repository")
	}
	metadata, blockCount, err := qwen35Metadata(repository)
	if err != nil {
		return model.Spec{}, err
	}
	mappings, err := qwen35TensorMappings(repository.Tensors, blockCount)
	if err != nil {
		return model.Spec{}, err
	}
	tensors, err := mappedTensorCatalog(mappings)
	if err != nil {
		return model.Spec{}, err
	}
	file := &gguf.File{Metadata: metadata, Tensors: tensors}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return model.Spec{}, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 model spec: %w", err)
	}
	if _, err := model.ReadWeights(file, spec); err != nil {
		return model.Spec{}, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 weight catalog: %w", err)
	}
	return spec, nil
}

// Qwen35TensorData: stream mapped Qwen 3.5 language payloads.
func Qwen35TensorData(repository *hfrepo.Repository) ([]gguf.TensorData, error) {
	if repository == nil || repository.Tensors == nil {
		return nil, errors.New("HF/GGUF adapter: nil repository")
	}
	text, err := qwen35TextConfig(repository.Config)
	if err != nil {
		return nil, err
	}
	blockCount, err := required[uint32](text, "num_hidden_layers")
	if err != nil {
		return nil, err
	}
	mappings, err := qwen35TensorMappings(repository.Tensors, blockCount)
	if err != nil {
		return nil, err
	}
	return mappedTensorData(mappings)
}

func qwen35Metadata(repository *hfrepo.Repository) ([]gguf.Metadata, uint32, error) {
	if repository.Identity.ModelType != "qwen3_5" || repository.Identity.TextModelType != "qwen3_5_text" {
		return nil, 0, fmt.Errorf(
			"HF/GGUF adapter: unsupported Qwen 3.5 identity %q/%q",
			repository.Identity.ModelType, repository.Identity.TextModelType,
		)
	}
	text, err := qwen35TextConfig(repository.Config)
	if err != nil {
		return nil, 0, err
	}
	contextLength, err := required[uint32](text, "max_position_embeddings")
	if err != nil {
		return nil, 0, err
	}
	embeddingLength, err := required[uint32](text, "hidden_size")
	if err != nil {
		return nil, 0, err
	}
	blockCount, err := required[uint32](text, "num_hidden_layers")
	if err != nil {
		return nil, 0, err
	}
	mtpBlocks, err := required[uint32](text, "mtp_num_hidden_layers")
	if err != nil {
		return nil, 0, err
	}
	if mtpBlocks != 1 || blockCount == math.MaxUint32 {
		return nil, 0, errors.New("HF/GGUF adapter: Qwen 3.5 requires one valid MTP layer")
	}
	feedForwardLength, err := required[uint32](text, "intermediate_size")
	if err != nil {
		return nil, 0, err
	}
	headCount, err := required[uint32](text, "num_attention_heads")
	if err != nil {
		return nil, 0, err
	}
	kvHeadCount, err := required[uint32](text, "num_key_value_heads")
	if err != nil {
		return nil, 0, err
	}
	headLength, err := required[uint32](text, "head_dim")
	if err != nil {
		return nil, 0, err
	}
	normEpsilon, err := required[float32](text, "rms_norm_eps")
	if err != nil {
		return nil, 0, err
	}
	vocabulary, err := required[uint32](text, "vocab_size")
	if err != nil {
		return nil, 0, err
	}
	fullAttentionInterval, err := required[uint32](text, "full_attention_interval")
	if err != nil {
		return nil, 0, err
	}
	convKernel, err := required[uint32](text, "linear_conv_kernel_dim")
	if err != nil {
		return nil, 0, err
	}
	stateSize, err := required[uint32](text, "linear_key_head_dim")
	if err != nil {
		return nil, 0, err
	}
	groupCount, err := required[uint32](text, "linear_num_key_heads")
	if err != nil {
		return nil, 0, err
	}
	timeStepRank, err := required[uint32](text, "linear_num_value_heads")
	if err != nil {
		return nil, 0, err
	}
	valueHeadLength, err := required[uint32](text, "linear_value_head_dim")
	if err != nil {
		return nil, 0, err
	}
	innerSize, err := checkedProduct(timeStepRank, valueHeadLength, "Qwen 3.5 recurrent inner size")
	if err != nil {
		return nil, 0, err
	}
	rope, err := required[map[string]json.RawMessage](text, "rope_parameters")
	if err != nil {
		return nil, 0, err
	}
	ropeBase, err := required[float32](rope, "rope_theta")
	if err != nil {
		return nil, 0, err
	}
	partialRotary, err := required[float64](rope, "partial_rotary_factor")
	if err != nil {
		return nil, 0, err
	}
	rotaryLength := float64(headLength) * partialRotary
	if rotaryLength <= 0 || rotaryLength > math.MaxUint32 || rotaryLength != math.Trunc(rotaryLength) {
		return nil, 0, errors.New("HF/GGUF adapter: Qwen 3.5 rotary dimension is invalid")
	}
	ropeSections, err := required[[]int32](rope, "mrope_section")
	if err != nil {
		return nil, 0, err
	}
	if len(ropeSections) == 3 {
		ropeSections = append(ropeSections, 0)
	}
	if len(ropeSections) != 4 {
		return nil, 0, errors.New("HF/GGUF adapter: Qwen 3.5 RoPE section count is invalid")
	}
	layerTypes, err := required[[]string](text, "layer_types")
	if err != nil {
		return nil, 0, err
	}
	if len(layerTypes) != int(blockCount) {
		return nil, 0, errors.New("HF/GGUF adapter: Qwen 3.5 layer type count is invalid")
	}
	recurrentLayers := make([]bool, len(layerTypes))
	for index, layerType := range layerTypes {
		switch layerType {
		case "linear_attention":
			recurrentLayers[index] = true
		case "full_attention":
		default:
			return nil, 0, fmt.Errorf("HF/GGUF adapter: unsupported Qwen 3.5 layer type %q", layerType)
		}
	}
	prefix := "qwen35."
	result := []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen35"),
		metadata("general.name", gguf.ValueTypeString, filepath.Base(repository.Directory)),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, blockCount+mtpBlocks),
		metadata(prefix+"nextn_predict_layers", gguf.ValueTypeUint32, mtpBlocks),
		metadata(prefix+"context_length", gguf.ValueTypeUint32, contextLength),
		metadata(prefix+"embedding_length", gguf.ValueTypeUint32, embeddingLength),
		metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, feedForwardLength),
		metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, headCount),
		metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, kvHeadCount),
		metadata(prefix+"attention.key_length", gguf.ValueTypeUint32, headLength),
		metadata(prefix+"attention.value_length", gguf.ValueTypeUint32, headLength),
		metadata(prefix+"rope.freq_base", gguf.ValueTypeFloat32, ropeBase),
		metadata(prefix+"rope.dimension_count", gguf.ValueTypeUint32, uint32(rotaryLength)),
		arrayMetadata(prefix+"rope.dimension_sections", gguf.ValueTypeInt32, ropeSections),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, normEpsilon),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, vocabulary),
		metadata(prefix+"ssm.conv_kernel", gguf.ValueTypeUint32, convKernel),
		metadata(prefix+"ssm.inner_size", gguf.ValueTypeUint32, innerSize),
		metadata(prefix+"ssm.state_size", gguf.ValueTypeUint32, stateSize),
		metadata(prefix+"ssm.time_step_rank", gguf.ValueTypeUint32, timeStepRank),
		metadata(prefix+"ssm.group_count", gguf.ValueTypeUint32, groupCount),
		metadata(prefix+"full_attention_interval", gguf.ValueTypeUint32, fullAttentionInterval),
		arrayMetadata(prefix+"attention.recurrent_layers", gguf.ValueTypeBool, recurrentLayers),
	}
	return result, blockCount, nil
}

func qwen35TextConfig(config map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	return required[map[string]json.RawMessage](config, "text_config")
}

func checkedProduct(left, right uint32, label string) (uint32, error) {
	if left == 0 || right == 0 || left > math.MaxUint32/right {
		return 0, fmt.Errorf("HF/GGUF adapter: %s is invalid", label)
	}
	return left * right, nil
}

func qwen35TensorMappings(source *safetensors.Source, blockCount uint32) ([]tensorMapping, error) {
	mappings := make([]tensorMapping, 0, len(source.Tensors))
	seen := make(map[string]string, len(source.Tensors))
	for _, sourceName := range source.Names() {
		tensor := source.Tensors[sourceName]
		name, include, err := qwen35TensorName(sourceName, blockCount)
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
		shape := tensor.Shape
		if strings.HasSuffix(sourceName, ".linear_attn.conv1d.weight") {
			if len(shape) != 3 || shape[1] != 1 {
				return nil, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 convolution tensor %q has shape %v", sourceName, shape)
			}
			shape = []uint64{shape[0], shape[2]}
		}
		if len(shape) == 0 || len(shape) > gguf.MaxDimensions {
			return nil, fmt.Errorf("HF/GGUF adapter: tensor %q rank %d is unsupported", sourceName, len(shape))
		}
		mappings = append(mappings, tensorMapping{name: name, tensor: tensor, shape: shape})
	}
	return mappings, nil
}

func qwen35TensorName(name string, blockCount uint32) (string, bool, error) {
	switch name {
	case "model.language_model.embed_tokens.weight":
		return "token_embd.weight", true, nil
	case "model.language_model.norm.weight":
		return "output_norm.weight", true, nil
	case "lm_head.weight":
		return "output.weight", true, nil
	case "mtp.fc.weight":
		return fmt.Sprintf("blk.%d.nextn.eh_proj.weight", blockCount), true, nil
	case "mtp.pre_fc_norm_embedding.weight":
		return fmt.Sprintf("blk.%d.nextn.enorm.weight", blockCount), true, nil
	case "mtp.pre_fc_norm_hidden.weight":
		return fmt.Sprintf("blk.%d.nextn.hnorm.weight", blockCount), true, nil
	case "mtp.norm.weight":
		return fmt.Sprintf("blk.%d.nextn.shared_head_norm.weight", blockCount), true, nil
	}
	if strings.HasPrefix(name, "model.visual.") {
		return "", false, nil
	}
	if rest, ok := strings.CutPrefix(name, "model.language_model.layers."); ok {
		return qwen35LayerTensorName(rest, 0, false)
	}
	if rest, ok := strings.CutPrefix(name, "mtp.layers."); ok {
		return qwen35LayerTensorName(rest, blockCount, true)
	}
	return "", false, fmt.Errorf("HF/GGUF adapter: tensor %q has no Qwen 3.5 mapping", name)
}

func qwen35LayerTensorName(rest string, blockOffset uint32, mtp bool) (string, bool, error) {
	layer, suffix, ok := strings.Cut(rest, ".")
	if !ok {
		return "", false, errors.New("HF/GGUF adapter: Qwen 3.5 tensor has no layer suffix")
	}
	index, err := strconv.ParseUint(layer, 10, 32)
	if err != nil {
		return "", false, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 layer %q is invalid", layer)
	}
	if mtp && index != 0 {
		return "", false, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 MTP layer %d is unsupported", index)
	}
	destination, ok := qwen35LayerNames[suffix]
	if !ok {
		return "", false, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 tensor suffix %q has no mapping", suffix)
	}
	return fmt.Sprintf("blk.%d.%s", uint64(blockOffset)+index, destination), true, nil
}

func arrayMetadata(key string, elementType gguf.ValueType, value any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{
		Type: gguf.ValueTypeArray, ArrayType: elementType, Data: value,
	}}
}
