package hfgguf

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"overgo/internal/safetensors"
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
	mappings, err := qwen35TensorMappings(repository, blockCount)
	if err != nil {
		return model.Spec{}, err
	}
	return validateMappedRepository(metadata, mappings, "Qwen 3.5")
}

// Qwen35Conversion: model metadata plus streamed language tensors for GGUF
// export; tokenizer metadata is the converter's responsibility.
func Qwen35Conversion(repository *hfrepo.Repository) ([]gguf.Metadata, []gguf.TensorData, error) {
	if repository == nil || repository.Tensors == nil {
		return nil, nil, errors.New("HF/GGUF adapter: nil repository")
	}
	metadata, blockCount, err := qwen35Metadata(repository)
	if err != nil {
		return nil, nil, err
	}
	mappings, err := qwen35TensorMappings(repository, blockCount)
	if err != nil {
		return nil, nil, err
	}
	tensors, err := mappedTensorData(mappings)
	if err != nil {
		return nil, nil, err
	}
	return metadata, tensors, nil
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
	dimensions, err := requiredValues[uint32](text,
		"max_position_embeddings", "hidden_size", "num_hidden_layers", "mtp_num_hidden_layers",
		"intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
		"full_attention_interval", "linear_conv_kernel_dim", "linear_key_head_dim",
		"linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
	)
	if err != nil {
		return nil, 0, err
	}
	contextLength, embeddingLength, blockCount, mtpBlocks := dimensions[0], dimensions[1], dimensions[2], dimensions[3]
	feedForwardLength, headCount, kvHeadCount, headLength := dimensions[4], dimensions[5], dimensions[6], dimensions[7]
	vocabulary, fullAttentionInterval, convKernel := dimensions[8], dimensions[9], dimensions[10]
	stateSize, groupCount, timeStepRank, valueHeadLength := dimensions[11], dimensions[12], dimensions[13], dimensions[14]
	if mtpBlocks != 1 || blockCount == math.MaxUint32 {
		return nil, 0, errors.New("HF/GGUF adapter: Qwen 3.5 requires one valid MTP layer")
	}
	normEpsilon, err := required[float32](text, "rms_norm_eps")
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
		gguf.ArrayMetadata(prefix+"rope.dimension_sections", gguf.ValueTypeInt32, ropeSections),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, normEpsilon),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, vocabulary),
		metadata(prefix+"ssm.conv_kernel", gguf.ValueTypeUint32, convKernel),
		metadata(prefix+"ssm.inner_size", gguf.ValueTypeUint32, innerSize),
		metadata(prefix+"ssm.state_size", gguf.ValueTypeUint32, stateSize),
		metadata(prefix+"ssm.time_step_rank", gguf.ValueTypeUint32, timeStepRank),
		metadata(prefix+"ssm.group_count", gguf.ValueTypeUint32, groupCount),
		metadata(prefix+"full_attention_interval", gguf.ValueTypeUint32, fullAttentionInterval),
		gguf.ArrayMetadata(prefix+"attention.recurrent_layers", gguf.ValueTypeBool, recurrentLayers),
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

func qwen35TensorMappings(repository *hfrepo.Repository, blockCount uint32) ([]tensorMapping, error) {
	dims, err := qwen35LinearDimensions(repository)
	if err != nil {
		return nil, err
	}
	return collectTensorMappings(repository.Tensors, func(name string) (string, bool, error) {
		return qwen35TensorName(name, blockCount)
	}, qwen35TensorShape, qwen35TensorTransforms(dims))
}

type qwen35LinearDims struct {
	keyHeads, valueHeads, keyWidth, valueWidth uint64
}

func qwen35LinearDimensions(repository *hfrepo.Repository) (qwen35LinearDims, error) {
	text, err := qwen35TextConfig(repository.Config)
	if err != nil {
		return qwen35LinearDims{}, err
	}
	values, err := requiredValues[uint32](text,
		"linear_num_key_heads", "linear_num_value_heads",
		"linear_key_head_dim", "linear_value_head_dim",
	)
	if err != nil {
		return qwen35LinearDims{}, err
	}
	dims := qwen35LinearDims{
		keyHeads: uint64(values[0]), valueHeads: uint64(values[1]),
		keyWidth: uint64(values[2]), valueWidth: uint64(values[3]),
	}
	if dims.keyHeads == 0 || dims.valueHeads%dims.keyHeads != 0 ||
		dims.keyWidth == 0 || dims.valueWidth == 0 {
		return qwen35LinearDims{}, errors.New("HF/GGUF adapter: Qwen 3.5 linear head dimensions are invalid")
	}
	return dims, nil
}

// qwen35TensorTransforms: llama.cpp GGUF conventions the runtime consumes —
// the GDN decay base is stored as -exp(A_log), and V-head-major payloads
// reorder from HF K-grouped order to tiled order when valueHeads > keyHeads.
func qwen35TensorTransforms(dims qwen35LinearDims) tensorValueMapper {
	reorder := dims.keyHeads != dims.valueHeads
	qkRows := 2 * dims.keyHeads * dims.keyWidth
	vRows := dims.valueHeads * dims.valueWidth
	return func(name string) func([]byte, uint64) error {
		if strings.HasSuffix(name, ".linear_attn.A_log") {
			return func(data []byte, elementSize uint64) error {
				if reorder {
					if err := dims.reorderVRows(data, elementSize, dims.valueHeads, 0, 1); err != nil {
						return err
					}
				}
				return negateExpF32(data, elementSize)
			}
		}
		// Zero-centered RMSNorm: every norm stores w, the runtime consumes 1+w;
		// the GDN group norm (linear_attn.norm) is conventional and stays raw.
		if strings.HasSuffix(name, "norm.weight") && !strings.HasSuffix(name, ".linear_attn.norm.weight") {
			return addOneF32
		}
		if !reorder {
			return nil
		}
		switch {
		case strings.HasSuffix(name, ".linear_attn.dt_bias"):
			return func(data []byte, elementSize uint64) error {
				return dims.reorderVRows(data, elementSize, dims.valueHeads, 0, 1)
			}
		case strings.HasSuffix(name, ".linear_attn.in_proj_qkv.weight"),
			strings.HasSuffix(name, ".linear_attn.conv1d.weight"):
			return func(data []byte, elementSize uint64) error {
				return dims.reorderVRows(data, elementSize, qkRows+vRows, qkRows, dims.valueWidth)
			}
		case strings.HasSuffix(name, ".linear_attn.in_proj_z.weight"):
			return func(data []byte, elementSize uint64) error {
				return dims.reorderVRows(data, elementSize, vRows, 0, dims.valueWidth)
			}
		case strings.HasSuffix(name, ".linear_attn.in_proj_a.weight"),
			strings.HasSuffix(name, ".linear_attn.in_proj_b.weight"):
			return func(data []byte, elementSize uint64) error {
				return dims.reorderVRows(data, elementSize, dims.valueHeads, 0, 1)
			}
		case strings.HasSuffix(name, ".linear_attn.out_proj.weight"):
			return func(data []byte, elementSize uint64) error {
				return dims.reorderVColumns(data, elementSize)
			}
		}
		return nil
	}
}

// reorderVRows: within the row region [startRow, startRow+valueHeads*blockRows),
// move row-blocks from K-grouped order (k*perK+v) to tiled order (v*keyHeads+k).
func (d qwen35LinearDims) reorderVRows(
	data []byte,
	elementSize, totalRows, startRow, blockRows uint64,
) error {
	perK := d.valueHeads / d.keyHeads
	totalBytes := uint64(len(data))
	if totalRows == 0 || totalBytes%(totalRows*elementSize) != 0 {
		return errors.New("V-row reorder: payload size disagrees with row count")
	}
	rowBytes := totalBytes / totalRows
	blockBytes := blockRows * rowBytes
	regionStart := startRow * rowBytes
	regionBytes := d.valueHeads * blockBytes
	if regionStart+regionBytes > totalBytes {
		return errors.New("V-row reorder: region exceeds payload")
	}
	region := data[regionStart : regionStart+regionBytes]
	scratch := make([]byte, regionBytes)
	for k := uint64(0); k < d.keyHeads; k++ {
		for v := uint64(0); v < perK; v++ {
			source := (k*perK + v) * blockBytes
			destination := (v*d.keyHeads + k) * blockBytes
			copy(scratch[destination:destination+blockBytes], region[source:source+blockBytes])
		}
	}
	copy(region, scratch)
	return nil
}

// reorderVColumns: per row, move valueWidth-wide column blocks from K-grouped
// to tiled V-head order (out_proj consumes the reordered V space as input).
func (d qwen35LinearDims) reorderVColumns(data []byte, elementSize uint64) error {
	perK := d.valueHeads / d.keyHeads
	headBytes := d.valueWidth * elementSize
	rowBytes := d.valueHeads * headBytes
	if rowBytes == 0 || uint64(len(data))%rowBytes != 0 {
		return errors.New("V-column reorder: payload size disagrees with row width")
	}
	scratch := make([]byte, rowBytes)
	for offset := uint64(0); offset < uint64(len(data)); offset += rowBytes {
		row := data[offset : offset+rowBytes]
		for k := uint64(0); k < d.keyHeads; k++ {
			for v := uint64(0); v < perK; v++ {
				source := (k*perK + v) * headBytes
				destination := (v*d.keyHeads + k) * headBytes
				copy(scratch[destination:destination+headBytes], row[source:source+headBytes])
			}
		}
		copy(row, scratch)
	}
	return nil
}

// addOneF32: zero-centered norm weights shift to 1+w on a promoted F32 payload.
func addOneF32(data []byte, elementSize uint64) error {
	if elementSize != 4 {
		return errors.New("norm transform requires an F32 payload")
	}
	for index := 0; index+4 <= len(data); index += 4 {
		value := math.Float32frombits(binary.LittleEndian.Uint32(data[index:]))
		binary.LittleEndian.PutUint32(data[index:], math.Float32bits(value+1))
	}
	return nil
}

// negateExpF32: decay base -exp(A_log) on a promoted F32 payload.
func negateExpF32(data []byte, elementSize uint64) error {
	if elementSize != 4 {
		return errors.New("A_log transform requires an F32 payload")
	}
	for index := 0; index+4 <= len(data); index += 4 {
		value := math.Float32frombits(binary.LittleEndian.Uint32(data[index:]))
		binary.LittleEndian.PutUint32(
			data[index:], math.Float32bits(float32(-math.Exp(float64(value)))),
		)
	}
	return nil
}

func qwen35TensorShape(tensor safetensors.Tensor) ([]uint64, error) {
	shape := tensor.Shape
	if !strings.HasSuffix(tensor.Name, ".linear_attn.conv1d.weight") {
		return shape, nil
	}
	if len(shape) != 3 || shape[1] != 1 {
		return nil, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 convolution tensor %q has shape %v", tensor.Name, shape)
	}
	return []uint64{shape[0], shape[2]}, nil
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
	if strings.HasPrefix(name, "model.language_model.layers.") {
		mapped, _, err := mapLayerTensor(name, "model.language_model.layers.", 0, qwen35LayerNames)
		return mapped, err == nil, err
	}
	if strings.HasPrefix(name, "mtp.layers.") {
		mapped, index, err := mapLayerTensor(name, "mtp.layers.", blockCount, qwen35LayerNames)
		if err == nil && index != 0 {
			err = fmt.Errorf("HF/GGUF adapter: Qwen 3.5 MTP layer %d is unsupported", index)
		}
		return mapped, err == nil, err
	}
	return "", false, fmt.Errorf("HF/GGUF adapter: tensor %q has no Qwen 3.5 mapping", name)
}
