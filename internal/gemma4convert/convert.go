package gemma4convert

import (
	"bytes"
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
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/projector"
	"overgo/internal/safetensors"
	"overgo/internal/tokenizer"
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
	} `json:"vision_config"`
	Audio struct {
		Embedding  uint32  `json:"audio_embed_dim"`
		RMSEpsilon float32 `json:"rms_norm_eps"`
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
	config, err := readConfig(directory)
	if err != nil {
		return Report{}, err
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
		tensors, tensorErr := modelTensors(source, config)
		if tensorErr != nil {
			return report, tensorErr
		}
		if err := validateModelCatalog(metadata, tensors); err != nil {
			return report, err
		}
		if err := writeOutput(options.ModelPath, metadata, tensors); err != nil {
			return report, err
		}
		report.ModelTensors = len(tensors)
	}
	if options.MMProjPath != "" {
		metadata := projectorMetadata(name, config)
		tensors, tensorErr := projectorTensors(source, options.MMProjF32)
		if tensorErr != nil {
			return report, tensorErr
		}
		if err := writeOutput(options.MMProjPath, metadata, tensors); err != nil {
			return report, err
		}
		runner, openErr := projector.OpenGemma4(options.MMProjPath)
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

func readConfig(directory string) (modelConfig, error) {
	var config modelConfig
	encoded, err := os.ReadFile(filepath.Join(directory, "config.json"))
	if err != nil {
		return config, fmt.Errorf("Gemma 4 converter: read config: %w", err)
	}
	if err := json.Unmarshal(encoded, &config); err != nil {
		return config, fmt.Errorf("Gemma 4 converter: parse config: %w", err)
	}
	return config, nil
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
	if config.Vision.PatchSize == 0 || config.Vision.PoolingSize == 0 || config.Vision.Embedding != text.HiddenSize ||
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
		stringMetadata("general.architecture", "gemma4"),
		stringMetadata("general.name", name),
		uint32Metadata("gemma4.block_count", text.HiddenLayers),
		uint32Metadata("gemma4.context_length", text.MaxPositions),
		uint32Metadata("gemma4.embedding_length", text.HiddenSize),
		arrayMetadata("gemma4.feed_forward_length", gguf.ValueTypeInt32, feedForward),
		uint32Metadata("gemma4.attention.head_count", text.AttentionHeads),
		arrayMetadata("gemma4.attention.head_count_kv", gguf.ValueTypeInt32, kvHeads),
		uint32Metadata("gemma4.attention.key_length", text.GlobalHeadDim),
		uint32Metadata("gemma4.attention.value_length", text.GlobalHeadDim),
		uint32Metadata("gemma4.attention.key_length_swa", text.HeadDim),
		uint32Metadata("gemma4.attention.value_length_swa", text.HeadDim),
		uint32Metadata("gemma4.rope.dimension_count", fullRope),
		uint32Metadata("gemma4.rope.dimension_count_swa", text.HeadDim),
		float32Metadata("gemma4.rope.freq_base", full.Theta),
		float32Metadata("gemma4.rope.freq_base_swa", sliding.Theta),
		uint32Metadata("gemma4.attention.sliding_window", text.SlidingWindow),
		arrayMetadata("gemma4.attention.sliding_window_pattern", gguf.ValueTypeBool, slidingLayers),
		uint32Metadata("gemma4.attention.shared_kv_layers", text.SharedKVLayers),
		uint32Metadata("gemma4.embedding_length_per_layer_input", text.PerLayerInput),
		float32Metadata("gemma4.attention.layer_norm_rms_epsilon", text.RMSEpsilon),
		float32Metadata("gemma4.final_logit_softcapping", text.FinalLogitSoftcap),
		uint32Metadata("gemma4.vocab_size", text.Vocabulary),
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
		stringMetadata("tokenizer.ggml.model", "gemma4"),
		stringMetadata("tokenizer.ggml.pre", "gemma4"),
		arrayMetadata("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		arrayMetadata("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
		arrayMetadata("tokenizer.ggml.merges", gguf.ValueTypeString, merges),
		uint32Metadata("tokenizer.ggml.bos_token_id", bos),
		uint32Metadata("tokenizer.ggml.eos_token_id", eos),
		uint32Metadata("tokenizer.ggml.eot_token_id", eot),
		uint32Metadata("tokenizer.ggml.unknown_token_id", unknown),
		uint32Metadata("tokenizer.ggml.padding_token_id", padding),
		uint32Metadata("tokenizer.ggml.mask_token_id", mask),
		boolMetadata("tokenizer.ggml.add_bos_token", true),
		boolMetadata("tokenizer.ggml.add_eos_token", false),
		boolMetadata("tokenizer.ggml.add_space_prefix", false),
	}
	if template, readErr := os.ReadFile(filepath.Join(directory, "chat_template.jinja")); readErr == nil {
		metadata = append(metadata, stringMetadata("tokenizer.chat_template", string(template)))
	} else if !errors.Is(readErr, os.ErrNotExist) {
		return nil, fmt.Errorf("Gemma 4 converter: read chat template: %w", readErr)
	}
	return metadata, nil
}

var byteTokenPattern = regexp.MustCompile(`^<0x[0-9A-Fa-f]{2}>$`)
var layerNamePattern = regexp.MustCompile(`^model\.language_model\.layers\.(\d+)\.(.+)$`)

func modelTensors(source *safetensors.Source, config modelConfig) ([]gguf.TensorData, error) {
	tensors := make([]gguf.TensorData, 0, len(source.Tensors)+1)
	for sourceName, tensor := range source.Tensors {
		destinationName, include := modelTensorName(sourceName)
		if !include {
			if strings.HasPrefix(sourceName, "model.language_model.") &&
				!strings.HasSuffix(sourceName, "weight_scale") {
				return nil, fmt.Errorf("Gemma 4 converter: language tensor %q has no mapping", sourceName)
			}
			continue
		}
		dataType, reader, err := modelTensorReader(source, tensor)
		if err != nil {
			return nil, fmt.Errorf("Gemma 4 converter: tensor %q: %w", sourceName, err)
		}
		tensors = append(tensors, gguf.TensorData{
			Name: destinationName, Shape: reverseShape(tensor.Shape), Type: dataType, Data: reader,
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

func modelTensorReader(source *safetensors.Source, tensor safetensors.Tensor) (gguf.DType, io.Reader, error) {
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
		reader, err := newFP8BF16Reader(tensor, scale)
		return gguf.DTypeBF16, reader, err
	default:
		return 0, nil, fmt.Errorf("unsupported dtype %q", tensor.DType)
	}
}

func validateModelCatalog(metadata []gguf.Metadata, tensors []gguf.TensorData) error {
	file := &gguf.File{Metadata: metadata, Tensors: make([]gguf.TensorInfo, len(tensors))}
	for index, tensor := range tensors {
		if len(tensor.Shape) > gguf.MaxDimensions {
			return fmt.Errorf("Gemma 4 converter: tensor %q rank exceeds GGUF", tensor.Name)
		}
		info := gguf.TensorInfo{
			Name: tensor.Name, Dimensions: uint32(len(tensor.Shape)), Type: tensor.Type,
			Shape: [gguf.MaxDimensions]uint64{1, 1, 1, 1},
		}
		copy(info.Shape[:], tensor.Shape)
		file.Tensors[index] = info
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return fmt.Errorf("Gemma 4 converter: generated model metadata: %w", err)
	}
	if _, err := model.ReadWeights(file, spec); err != nil {
		return fmt.Errorf("Gemma 4 converter: generated tensor catalog: %w", err)
	}
	if _, err := tokenizer.Load(file); err != nil {
		return fmt.Errorf("Gemma 4 converter: generated tokenizer: %w", err)
	}
	return nil
}

func projectorMetadata(name string, config modelConfig) []gguf.Metadata {
	return []gguf.Metadata{
		stringMetadata("general.architecture", "clip"),
		stringMetadata("general.name", name+" multimodal projector"),
		stringMetadata("clip.vision.projector_type", "gemma4uv"),
		boolMetadata("clip.has_vision_encoder", true),
		uint32Metadata("clip.vision.image_size", gemma4ProjectorImageSize),
		uint32Metadata("clip.vision.patch_size", config.Vision.PatchSize),
		uint32Metadata("clip.vision.embedding_length", config.Vision.Embedding),
		uint32Metadata("clip.vision.feed_forward_length", 0),
		uint32Metadata("clip.vision.block_count", 0),
		uint32Metadata("clip.vision.attention.head_count", 0),
		uint32Metadata("clip.vision.projection_dim", config.Vision.Embedding),
		float32Metadata("clip.vision.attention.layer_norm_epsilon", config.Vision.RMSEpsilon),
		arrayMetadata("clip.vision.image_mean", gguf.ValueTypeFloat32, []float32{0, 0, 0}),
		arrayMetadata("clip.vision.image_std", gguf.ValueTypeFloat32, []float32{1, 1, 1}),
		uint32Metadata("clip.vision.projector_scale_factor", config.Vision.PoolingSize),
		stringMetadata("clip.audio.projector_type", "gemma4ua"),
		boolMetadata("clip.has_audio_encoder", true),
		uint32Metadata("clip.audio.embedding_length", config.Audio.Embedding),
		uint32Metadata("clip.audio.feed_forward_length", 0),
		uint32Metadata("clip.audio.block_count", 0),
		uint32Metadata("clip.audio.attention.head_count", 0),
		uint32Metadata("clip.audio.projection_dim", config.Text.HiddenSize),
		float32Metadata("clip.audio.attention.layer_norm_epsilon", config.Audio.RMSEpsilon),
		uint32Metadata("clip.audio.num_mel_bins", 128),
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
		shape := reverseShape(tensor.Shape)
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

func writeOutput(path string, metadata []gguf.Metadata, tensors []gguf.TensorData) error {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(absolute, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return fmt.Errorf("Gemma 4 converter: create %s: %w", absolute, err)
	}
	succeeded := false
	defer func() {
		_ = file.Close()
		if !succeeded {
			_ = os.Remove(absolute)
		}
	}()
	if err := gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		return fmt.Errorf("Gemma 4 converter: write %s: %w", absolute, err)
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	succeeded = true
	return nil
}

func reverseShape(shape []uint64) []uint64 {
	result := make([]uint64, len(shape))
	for index := range shape {
		result[len(shape)-1-index] = shape[index]
	}
	return result
}

func stringMetadata(key, value string) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: gguf.ValueTypeString, Data: value}}
}

func uint32Metadata(key string, value uint32) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: value}}
}

func float32Metadata(key string, value float32) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: value}}
}

func boolMetadata(key string, value bool) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: gguf.ValueTypeBool, Data: value}}
}

func arrayMetadata(key string, valueType gguf.ValueType, value any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: valueType, Data: value}}
}
