package gemma4convert

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/jsonfile"
	"overgo/internal/modelartifact"
	"overgo/internal/projector"
	"overgo/internal/safetensors"
)

const gemma4ProjectorImageSize = 224

type modelConfig struct {
	Text struct {
		FinalLogitSoftcap float32  `json:"final_logit_softcapping"`
		GlobalHeadDim     uint32   `json:"global_head_dim"`
		HeadDim           uint32   `json:"head_dim"`
		HiddenSize        uint32   `json:"hidden_size"`
		IntermediateSize  uint32   `json:"intermediate_size"`
		LayerTypes        []string `json:"layer_types"`
		MaxPositions      uint32   `json:"max_position_embeddings"`
		AttentionHeads    uint32   `json:"num_attention_heads"`
		GlobalKVHeads     uint32   `json:"num_global_key_value_heads"`
		HiddenLayers      uint32   `json:"num_hidden_layers"`
		KVHeads           uint32   `json:"num_key_value_heads"`
		SharedKVLayers    uint32   `json:"num_kv_shared_layers"`
		RMSEpsilon        float32  `json:"rms_norm_eps"`
		Rope              map[string]struct {
			PartialRotary float32 `json:"partial_rotary_factor"`
			Theta         float32 `json:"rope_theta"`
			Type          string  `json:"rope_type"`
		} `json:"rope_parameters"`
		SlidingWindow uint32 `json:"sliding_window"`
		Vocabulary    uint32 `json:"vocab_size"`
		PerLayerInput uint32 `json:"hidden_size_per_layer_input"`
	} `json:"text_config"`
	Vision struct {
		Embedding   uint32  `json:"mm_embed_dim"`
		Positions   uint32  `json:"mm_posemb_size"`
		PatchSize   uint32  `json:"patch_size"`
		PoolingSize uint32  `json:"pooling_kernel_size"`
		RMSEpsilon  float32 `json:"rms_norm_eps"`
		// tower layout (E4B): full vision transformer
		HiddenSize        uint32 `json:"hidden_size"`
		HiddenLayers      uint32 `json:"num_hidden_layers"`
		AttentionHeads    uint32 `json:"num_attention_heads"`
		KVHeads           uint32 `json:"num_key_value_heads"`
		HeadDim           uint32 `json:"head_dim"`
		IntermediateSize  uint32 `json:"intermediate_size"`
		PositionEmbedding uint32 `json:"position_embedding_size"`
		HiddenAct         string `json:"hidden_activation"`
		Rope              struct {
			Theta float32 `json:"rope_theta"`
		} `json:"rope_parameters"`
	} `json:"vision_config"`
	Audio struct {
		Embedding  uint32  `json:"audio_embed_dim"`
		RMSEpsilon float32 `json:"rms_norm_eps"`
		// tower layout (E4B): conformer encoder
		HiddenSize     uint32   `json:"hidden_size"`
		HiddenLayers   uint32   `json:"num_hidden_layers"`
		AttentionHeads uint32   `json:"num_attention_heads"`
		ConvKernel     uint32   `json:"conv_kernel_size"`
		SubChannels    []uint32 `json:"subsampling_conv_channels"`
		ChunkSize      uint32   `json:"attention_chunk_size"`
		ContextLeft    uint32   `json:"attention_context_left"`
		ContextRight   uint32   `json:"attention_context_right"`
		LogitCap       float32  `json:"attention_logit_cap"`
		ResidualWeight float32  `json:"residual_weight"`
		OutputProjDims uint32   `json:"output_proj_dims"`
		HiddenAct      string   `json:"hidden_act"`
	} `json:"audio_config"`
}

type tokenizerFile struct {
	Added []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges [][]string     `json:"merges"`
	} `json:"model"`
}

// Options: one Gemma 4 checkpoint export.
type Options struct {
	Directory  string
	ModelPath  string
	MMProjPath string
	MMProjF32  bool
	Name       string
	// FP8Native: preserve fp8 (F8_E4M3) mlp gate/up/down weights as native
	// F8E4M3 GGUF tensors ([e4m3 | per-row F32 scale] combined payload) instead
	// of up-converting them to BF16. attn/lm_head/embeds always stay BF16.
	FP8Native bool
}

type Report struct {
	ModelTensors  int
	MMProjTensors int
}

func Convert(options Options) (Report, error) {
	directory, err := filepath.Abs(options.Directory)
	if err != nil {
		return Report{}, err
	}
	if strings.TrimSpace(options.ModelPath) == "" && strings.TrimSpace(options.MMProjPath) == "" {
		return Report{}, errors.New("Gemma 4 converter: model or mmproj output is required")
	}
	for _, output := range []*string{&options.ModelPath, &options.MMProjPath} {
		if *output == "" {
			continue
		}
		absolute, pathErr := filepath.Abs(*output)
		if pathErr != nil {
			return Report{}, pathErr
		}
		*output = absolute
	}
	if options.ModelPath != "" && options.MMProjPath != "" &&
		strings.EqualFold(filepath.Clean(options.ModelPath), filepath.Clean(options.MMProjPath)) {
		return Report{}, errors.New("Gemma 4 converter: model and mmproj outputs are identical")
	}
	var config modelConfig
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &config); err != nil {
		return Report{}, fmt.Errorf("Gemma 4 converter: config: %w", err)
	}
	if config.Text.GlobalKVHeads == 0 {
		// null num_global_key_value_heads: global layers reuse num_key_value_heads (E4B)
		config.Text.GlobalKVHeads = config.Text.KVHeads
	}
	if err := validateConfig(config); err != nil {
		return Report{}, err
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return Report{}, err
	}
	defer source.Close()
	name := strings.TrimSpace(options.Name)
	if name == "" {
		name = filepath.Base(directory)
	}
	report := Report{}
	if options.ModelPath != "" {
		metadata, metadataErr := modelMetadata(directory, name, config)
		if metadataErr != nil {
			return report, metadataErr
		}
		tensors, tensorErr := modelTensors(source, config, options.FP8Native)
		if tensorErr != nil {
			return report, tensorErr
		}
		if err := modelartifact.ValidateGeneratedCatalog(metadata, tensors); err != nil {
			return report, err
		}
		if err := gguf.WriteFileExclusive(options.ModelPath, metadata, tensors, gguf.WriteOptions{}); err != nil {
			return report, err
		}
		report.ModelTensors = len(tensors)
	}
	if options.MMProjPath != "" {
		tower := towerLayout(config)
		var metadata []gguf.Metadata
		var tensors []gguf.TensorData
		if tower {
			var processor processorConfig
			if err := jsonfile.Decode(filepath.Join(directory, "processor_config.json"), &processor); err != nil {
				return report, fmt.Errorf("Gemma 4 converter: processor config: %w", err)
			}
			if err := validateTowerConfig(config, processor); err != nil {
				return report, err
			}
			feedForward, feedErr := audioFeedForwardLength(source, config)
			if feedErr != nil {
				return report, feedErr
			}
			metadata = towerProjectorMetadata(name, config, processor, feedForward)
			var tensorErr error
			tensors, tensorErr = towerTensors(source, options.MMProjF32)
			if tensorErr != nil {
				return report, tensorErr
			}
		} else {
			if err := validateProjectorConfig(config); err != nil {
				return report, err
			}
			metadata = projectorMetadata(name, config)
			var tensorErr error
			tensors, tensorErr = projectorTensors(source, options.MMProjF32)
			if tensorErr != nil {
				return report, tensorErr
			}
		}
		if err := gguf.WriteFileExclusive(options.MMProjPath, metadata, tensors, gguf.WriteOptions{}); err != nil {
			return report, err
		}
		runner, openErr := projector.OpenAs[projector.Projector](context.Background(), options.MMProjPath, projector.OpenOptions{})
		if openErr != nil {
			_ = os.Remove(options.MMProjPath)
			return report, fmt.Errorf("Gemma 4 converter: generated projector: %w", openErr)
		}
		if closeErr := runner.Close(); closeErr != nil {
			return report, fmt.Errorf("Gemma 4 converter: close generated projector: %w", closeErr)
		}
		report.MMProjTensors = len(tensors)
	}
	return report, nil
}

func validateConfig(config modelConfig) error {
	text := config.Text
	if text.HiddenLayers == 0 || int(text.HiddenLayers) != len(text.LayerTypes) ||
		text.HiddenSize == 0 || text.IntermediateSize == 0 || text.AttentionHeads == 0 ||
		text.GlobalHeadDim == 0 || text.HeadDim == 0 || text.GlobalKVHeads == 0 ||
		text.KVHeads == 0 || text.Vocabulary == 0 || text.MaxPositions == 0 {
		return errors.New("Gemma 4 converter: text configuration is incomplete")
	}
	for _, layerType := range text.LayerTypes {
		if layerType != "sliding_attention" && layerType != "full_attention" {
			return fmt.Errorf("Gemma 4 converter: unsupported layer type %q", layerType)
		}
	}
	full, fullOK := text.Rope["full_attention"]
	sliding, slidingOK := text.Rope["sliding_attention"]
	if !fullOK || !slidingOK || full.Type != "proportional" || full.Theta <= 0 || sliding.Theta <= 0 ||
		full.PartialRotary <= 0 || full.PartialRotary > 1 {
		return errors.New("Gemma 4 converter: RoPE configuration is incomplete")
	}
	return nil
}

// validateProjectorConfig: embedder-layout multimodal keys; only the mmproj output needs them.
func validateProjectorConfig(config modelConfig) error {
	if config.Vision.PatchSize == 0 || config.Vision.PoolingSize == 0 || config.Vision.Embedding != config.Text.HiddenSize ||
		config.Vision.Positions == 0 || config.Audio.Embedding == 0 {
		return errors.New("Gemma 4 converter: multimodal configuration is incomplete")
	}
	return nil
}

func modelMetadata(directory, name string, config modelConfig) ([]gguf.Metadata, error) {
	text := config.Text
	full := text.Rope["full_attention"]
	sliding := text.Rope["sliding_attention"]
	fullRope := text.GlobalHeadDim
	if fullRope == 0 || fullRope%2 != 0 || text.HeadDim%2 != 0 {
		return nil, errors.New("Gemma 4 converter: rotary dimensions are invalid")
	}
	feedForward := make([]int32, text.HiddenLayers)
	kvHeads := make([]int32, text.HiddenLayers)
	slidingLayers := make([]bool, text.HiddenLayers)
	for index, layerType := range text.LayerTypes {
		feedForward[index] = int32(text.IntermediateSize)
		slidingLayers[index] = layerType == "sliding_attention"
		if slidingLayers[index] {
			kvHeads[index] = int32(text.KVHeads)
		} else {
			kvHeads[index] = int32(text.GlobalKVHeads)
		}
	}
	metadata := []gguf.Metadata{
		gguf.StringMetadata("general.architecture", "gemma4"),
		gguf.StringMetadata("general.name", name),
		gguf.Uint32Metadata("gemma4.block_count", text.HiddenLayers),
		gguf.Uint32Metadata("gemma4.context_length", text.MaxPositions),
		gguf.Uint32Metadata("gemma4.embedding_length", text.HiddenSize),
		gguf.ArrayMetadata("gemma4.feed_forward_length", gguf.ValueTypeInt32, feedForward),
		gguf.Uint32Metadata("gemma4.attention.head_count", text.AttentionHeads),
		gguf.ArrayMetadata("gemma4.attention.head_count_kv", gguf.ValueTypeInt32, kvHeads),
		gguf.Uint32Metadata("gemma4.attention.key_length", text.GlobalHeadDim),
		gguf.Uint32Metadata("gemma4.attention.value_length", text.GlobalHeadDim),
		gguf.Uint32Metadata("gemma4.attention.key_length_swa", text.HeadDim),
		gguf.Uint32Metadata("gemma4.attention.value_length_swa", text.HeadDim),
		gguf.Uint32Metadata("gemma4.rope.dimension_count", fullRope),
		gguf.Uint32Metadata("gemma4.rope.dimension_count_swa", text.HeadDim),
		gguf.Float32Metadata("gemma4.rope.freq_base", full.Theta),
		gguf.Float32Metadata("gemma4.rope.freq_base_swa", sliding.Theta),
		gguf.Uint32Metadata("gemma4.attention.sliding_window", text.SlidingWindow),
		gguf.ArrayMetadata("gemma4.attention.sliding_window_pattern", gguf.ValueTypeBool, slidingLayers),
		gguf.Uint32Metadata("gemma4.attention.shared_kv_layers", text.SharedKVLayers),
		gguf.Uint32Metadata("gemma4.embedding_length_per_layer_input", text.PerLayerInput),
		gguf.Float32Metadata("gemma4.attention.layer_norm_rms_epsilon", text.RMSEpsilon),
		gguf.Float32Metadata("gemma4.final_logit_softcapping", text.FinalLogitSoftcap),
		gguf.Uint32Metadata("gemma4.vocab_size", text.Vocabulary),
	}
	tokenizer, err := tokenizerMetadata(directory, text.Vocabulary)
	if err != nil {
		return nil, err
	}
	return append(metadata, tokenizer...), nil
}

func tokenizerMetadata(directory string, vocabulary uint32) ([]gguf.Metadata, error) {
	encoded, err := os.ReadFile(filepath.Join(directory, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("Gemma 4 converter: read tokenizer: %w", err)
	}
	var tokenizer tokenizerFile
	if err := json.Unmarshal(encoded, &tokenizer); err != nil {
		return nil, fmt.Errorf("Gemma 4 converter: parse tokenizer: %w", err)
	}
	if len(tokenizer.Model.Vocab) != int(vocabulary) {
		return nil, fmt.Errorf("Gemma 4 converter: tokenizer has %d tokens, need %d", len(tokenizer.Model.Vocab), vocabulary)
	}
	tokens := make([]string, vocabulary)
	seen := make([]bool, vocabulary)
	for token, id := range tokenizer.Model.Vocab {
		if id < 0 || id >= len(tokens) || seen[id] {
			return nil, fmt.Errorf("Gemma 4 converter: invalid tokenizer ID %d", id)
		}
		tokens[id], seen[id] = token, true
	}
	for id, present := range seen {
		if !present {
			return nil, fmt.Errorf("Gemma 4 converter: tokenizer ID %d is missing", id)
		}
	}
	types := make([]int32, len(tokens))
	for index, token := range tokens {
		types[index] = 1
		if byteTokenPattern.MatchString(token) {
			types[index] = 6
		}
	}
	for _, added := range tokenizer.Added {
		if added.ID < 0 || added.ID >= len(tokens) || tokens[added.ID] != added.Content {
			return nil, fmt.Errorf("Gemma 4 converter: added token ID %d is inconsistent", added.ID)
		}
		if added.Special {
			types[added.ID] = 3
		}
	}
	if id, ok := tokenizer.Model.Vocab["<unk>"]; ok {
		types[id] = 2
	}
	merges := make([]string, len(tokenizer.Model.Merges))
	for index, pair := range tokenizer.Model.Merges {
		if len(pair) != 2 || pair[0] == "" || pair[1] == "" {
			return nil, fmt.Errorf("Gemma 4 converter: merge %d is invalid", index)
		}
		merges[index] = pair[0] + " " + pair[1]
	}
	requiredID := func(token string) (uint32, error) {
		id, ok := tokenizer.Model.Vocab[token]
		if !ok {
			return 0, fmt.Errorf("Gemma 4 converter: token %q is missing", token)
		}
		return uint32(id), nil
	}
	bos, err := requiredID("<bos>")
	if err != nil {
		return nil, err
	}
	eos, err := requiredID("<eos>")
	if err != nil {
		return nil, err
	}
	unknown, err := requiredID("<unk>")
	if err != nil {
		return nil, err
	}
	padding, err := requiredID("<pad>")
	if err != nil {
		return nil, err
	}
	mask, err := requiredID("<mask>")
	if err != nil {
		return nil, err
	}
	eot, err := requiredID("<turn|>")
	if err != nil {
		return nil, err
	}
	metadata := []gguf.Metadata{
		gguf.StringMetadata("tokenizer.ggml.model", "gemma4"),
		gguf.StringMetadata("tokenizer.ggml.pre", "gemma4"),
		gguf.ArrayMetadata("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		gguf.ArrayMetadata("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
		gguf.ArrayMetadata("tokenizer.ggml.merges", gguf.ValueTypeString, merges),
		gguf.Uint32Metadata("tokenizer.ggml.bos_token_id", bos),
		gguf.Uint32Metadata("tokenizer.ggml.eos_token_id", eos),
		gguf.Uint32Metadata("tokenizer.ggml.eot_token_id", eot),
		gguf.Uint32Metadata("tokenizer.ggml.unknown_token_id", unknown),
		gguf.Uint32Metadata("tokenizer.ggml.padding_token_id", padding),
		gguf.Uint32Metadata("tokenizer.ggml.mask_token_id", mask),
		gguf.BoolMetadata("tokenizer.ggml.add_bos_token", true),
		gguf.BoolMetadata("tokenizer.ggml.add_eos_token", false),
		gguf.BoolMetadata("tokenizer.ggml.add_space_prefix", false),
	}
	if template, readErr := os.ReadFile(filepath.Join(directory, "chat_template.jinja")); readErr == nil {
		metadata = append(metadata, gguf.StringMetadata("tokenizer.chat_template", string(template)))
	} else if !errors.Is(readErr, os.ErrNotExist) {
		return nil, fmt.Errorf("Gemma 4 converter: read chat template: %w", readErr)
	}
	return metadata, nil
}

var byteTokenPattern = regexp.MustCompile(`^<0x[0-9A-Fa-f]{2}>$`)
var layerNamePattern = regexp.MustCompile(`^model\.language_model\.layers\.(\d+)\.(.+)$`)

func modelTensors(source *safetensors.Source, config modelConfig, fp8Native bool) ([]gguf.TensorData, error) {
	tensors := make([]gguf.TensorData, 0, len(source.Tensors)+1)
	sharedKVStart := config.Text.HiddenLayers - config.Text.SharedKVLayers
	for sourceName, tensor := range source.Tensors {
		destinationName, include := modelTensorName(sourceName)
		if include && config.Text.SharedKVLayers > 0 &&
			(strings.HasSuffix(destinationName, ".attn_k.weight") || strings.HasSuffix(destinationName, ".attn_v.weight")) {
			match := layerNamePattern.FindStringSubmatch(sourceName)
			if index, parseErr := strconv.ParseUint(match[1], 10, 32); parseErr == nil && uint32(index) >= sharedKVStart {
				// shared-KV blocks never read k/v projections (spec.LayerHasKV)
				continue
			}
		}
		if !include {
			if strings.HasPrefix(sourceName, "model.language_model.") &&
				!strings.HasSuffix(sourceName, "weight_scale") {
				return nil, fmt.Errorf("Gemma 4 converter: language tensor %q has no mapping", sourceName)
			}
			continue
		}
		dataType, reader, err := modelTensorReader(source, tensor, destinationName, fp8Native)
		if err != nil {
			return nil, fmt.Errorf("Gemma 4 converter: tensor %q: %w", sourceName, err)
		}
		tensors = append(tensors, gguf.TensorData{
			Name: destinationName, Shape: gguf.ReverseShape(tensor.Shape), Type: dataType, Data: reader,
		})
	}
	tensors = append(tensors, proportionalRopeTensor(config))
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })
	if len(tensors) == 0 {
		return nil, errors.New("Gemma 4 converter: no language tensors found")
	}
	return tensors, nil
}

func proportionalRopeTensor(config modelConfig) gguf.TensorData {
	full := config.Text.Rope["full_attention"]
	pairs := config.Text.GlobalHeadDim / 2
	rotated := uint32(float32(config.Text.GlobalHeadDim)*full.PartialRotary) / 2
	encoded := make([]byte, pairs*4)
	for index := uint32(0); index < pairs; index++ {
		factor := float32(1)
		if index >= rotated {
			factor = 1e30
		}
		binary.LittleEndian.PutUint32(encoded[index*4:], math.Float32bits(factor))
	}
	return gguf.TensorData{
		Name: "rope_freqs.weight", Shape: []uint64{uint64(pairs)}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(encoded),
	}
}

func modelTensorName(name string) (string, bool) {
	switch name {
	case "model.language_model.embed_tokens.weight":
		return "token_embd.weight", true
	case "model.language_model.norm.weight":
		return "output_norm.weight", true
	case "model.language_model.embed_tokens_per_layer.weight":
		return "per_layer_token_embd.weight", true
	case "model.language_model.per_layer_model_projection.weight":
		return "per_layer_model_proj.weight", true
	case "model.language_model.per_layer_projection_norm.weight":
		return "per_layer_proj_norm.weight", true
	}
	match := layerNamePattern.FindStringSubmatch(name)
	if match == nil || strings.HasSuffix(name, "weight_scale") {
		return "", false
	}
	mapping := map[string]string{
		"input_layernorm.weight":            "attn_norm.weight",
		"layer_scalar":                      "layer_output_scale.weight",
		"mlp.down_proj.weight":              "ffn_down.weight",
		"mlp.gate_proj.weight":              "ffn_gate.weight",
		"mlp.up_proj.weight":                "ffn_up.weight",
		"per_layer_input_gate.weight":       "per_layer_inp_gate.weight",
		"per_layer_projection.weight":       "per_layer_proj.weight",
		"post_per_layer_input_norm.weight":  "per_layer_post_norm.weight",
		"post_attention_layernorm.weight":   "post_attention_norm.weight",
		"post_feedforward_layernorm.weight": "post_ffw_norm.weight",
		"pre_feedforward_layernorm.weight":  "ffn_norm.weight",
		"self_attn.k_norm.weight":           "attn_k_norm.weight",
		"self_attn.k_proj.weight":           "attn_k.weight",
		"self_attn.o_proj.weight":           "attn_output.weight",
		"self_attn.q_norm.weight":           "attn_q_norm.weight",
		"self_attn.q_proj.weight":           "attn_q.weight",
		"self_attn.v_proj.weight":           "attn_v.weight",
	}
	suffix, ok := mapping[match[2]]
	if !ok {
		return "", false
	}
	return "blk." + match[1] + "." + suffix, true
}

func modelTensorReader(
	source *safetensors.Source, tensor safetensors.Tensor, destination string, fp8Native bool,
) (gguf.DType, io.Reader, error) {
	if strings.HasSuffix(tensor.Name, ".layer_scalar") {
		if tensor.DType != "BF16" {
			return 0, nil, errors.New("layer scalar must use BF16")
		}
		reader, err := safetensors.F32Reader(tensor)
		return gguf.DTypeF32, reader, err
	}
	switch tensor.DType {
	case "BF16":
		return gguf.DTypeBF16, tensor.Reader(), nil
	case "F32":
		return gguf.DTypeF32, tensor.Reader(), nil
	case "F8_E4M3":
		scale, ok := source.Tensors[tensor.Name+"_scale"]
		if !ok {
			return 0, nil, errors.New("FP8 scale is missing")
		}
		if fp8Native && isFFNProjection(destination) {
			// native fp8 residency: preserve [e4m3 | per-row F32 scale] combined
			reader, err := newFP8NativeReader(tensor, scale)
			return gguf.DTypeF8E4M3, reader, err
		}
		reader, err := newFP8BF16Reader(tensor, scale)
		return gguf.DTypeBF16, reader, err
	default:
		return 0, nil, fmt.Errorf("unsupported dtype %q", tensor.DType)
	}
}

// isFFNProjection: gemma4 mlp gate/up/down destination weights -- the only
// tensors emitted natively as F8E4M3.
func isFFNProjection(destination string) bool {
	return strings.HasSuffix(destination, ".ffn_gate.weight") ||
		strings.HasSuffix(destination, ".ffn_up.weight") ||
		strings.HasSuffix(destination, ".ffn_down.weight")
}

func projectorMetadata(name string, config modelConfig) []gguf.Metadata {
	return []gguf.Metadata{
		gguf.StringMetadata("general.architecture", "clip"),
		gguf.StringMetadata("general.name", name+" multimodal projector"),
		gguf.StringMetadata("clip.vision.projector_type", "gemma4uv"),
		gguf.BoolMetadata("clip.has_vision_encoder", true),
		gguf.Uint32Metadata("clip.vision.image_size", gemma4ProjectorImageSize),
		gguf.Uint32Metadata("clip.vision.patch_size", config.Vision.PatchSize),
		gguf.Uint32Metadata("clip.vision.embedding_length", config.Vision.Embedding),
		gguf.Uint32Metadata("clip.vision.feed_forward_length", 0),
		gguf.Uint32Metadata("clip.vision.block_count", 0),
		gguf.Uint32Metadata("clip.vision.attention.head_count", 0),
		gguf.Uint32Metadata("clip.vision.projection_dim", config.Vision.Embedding),
		gguf.Float32Metadata("clip.vision.attention.layer_norm_epsilon", config.Vision.RMSEpsilon),
		gguf.ArrayMetadata("clip.vision.image_mean", gguf.ValueTypeFloat32, []float32{0, 0, 0}),
		gguf.ArrayMetadata("clip.vision.image_std", gguf.ValueTypeFloat32, []float32{1, 1, 1}),
		gguf.Uint32Metadata("clip.vision.projector_scale_factor", config.Vision.PoolingSize),
		gguf.StringMetadata("clip.audio.projector_type", "gemma4ua"),
		gguf.BoolMetadata("clip.has_audio_encoder", true),
		gguf.Uint32Metadata("clip.audio.embedding_length", config.Audio.Embedding),
		gguf.Uint32Metadata("clip.audio.feed_forward_length", 0),
		gguf.Uint32Metadata("clip.audio.block_count", 0),
		gguf.Uint32Metadata("clip.audio.attention.head_count", 0),
		gguf.Uint32Metadata("clip.audio.projection_dim", config.Text.HiddenSize),
		gguf.Float32Metadata("clip.audio.attention.layer_norm_epsilon", config.Audio.RMSEpsilon),
		gguf.Uint32Metadata("clip.audio.num_mel_bins", 128),
	}
}

func projectorTensors(source *safetensors.Source, outputF32 bool) ([]gguf.TensorData, error) {
	mapping := []struct{ source, destination string }{
		{"model.embed_vision.patch_dense.weight", "v.patch_embd.weight"},
		{"model.embed_vision.patch_dense.bias", "v.patch_embd.bias"},
		{"model.embed_vision.patch_ln1.weight", "v.patch_norm.1.weight"},
		{"model.embed_vision.patch_ln1.bias", "v.patch_norm.1.bias"},
		{"model.embed_vision.patch_ln2.weight", "v.patch_norm.2.weight"},
		{"model.embed_vision.patch_ln2.bias", "v.patch_norm.2.bias"},
		{"model.embed_vision.pos_embedding", "v.position_embd.weight"},
		{"model.embed_vision.pos_norm.weight", "v.patch_norm.3.weight"},
		{"model.embed_vision.pos_norm.bias", "v.patch_norm.3.bias"},
		{"model.embed_vision.multimodal_embedder.embedding_projection.weight", "mm.input_projection.weight"},
		{"model.embed_audio.embedding_projection.weight", "mm.a.input_projection.weight"},
	}
	tensors := make([]gguf.TensorData, 0, len(mapping))
	for _, item := range mapping {
		tensor, ok := source.Tensors[item.source]
		if !ok {
			return nil, fmt.Errorf("Gemma 4 converter: projector tensor %q is missing", item.source)
		}
		if tensor.DType != "BF16" {
			return nil, fmt.Errorf("Gemma 4 converter: projector tensor %q uses %s", item.source, tensor.DType)
		}
		shape := gguf.ReverseShape(tensor.Shape)
		var reader io.Reader = tensor.Reader()
		switch item.source {
		case "model.embed_vision.patch_dense.weight",
			"model.embed_vision.patch_ln1.weight",
			"model.embed_vision.patch_ln1.bias":
			permuted, err := newPatchPermutationReader(tensor)
			if err != nil {
				return nil, fmt.Errorf("Gemma 4 converter: %w", err)
			}
			reader = permuted
		case "model.embed_vision.pos_embedding":
			position, err := newPositionReader(tensor)
			if err != nil {
				return nil, fmt.Errorf("Gemma 4 converter: %w", err)
			}
			reader = position
			shape = []uint64{tensor.Shape[2], tensor.Shape[0], tensor.Shape[1]}
		}
		dataType := gguf.DTypeBF16
		if outputF32 {
			dataType = gguf.DTypeF32
			promoted, promoteErr := safetensors.PromoteF32Reader(reader, "BF16")
			if promoteErr != nil {
				return nil, promoteErr
			}
			reader = promoted
		}
		tensors = append(tensors, gguf.TensorData{
			Name: item.destination, Shape: shape, Type: dataType, Data: reader,
		})
	}
	return tensors, nil
}
