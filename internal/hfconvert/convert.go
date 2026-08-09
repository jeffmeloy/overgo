// Package hfconvert converts a plain HF-llama safetensors checkpoint to GGUF.
// Dims and vocabulary derive entirely from config.json and tokenizer files.
package hfconvert

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/safetensors"
	"overgo/internal/tokenizer"
)

type modelConfig struct {
	ModelType        string  `json:"model_type"`
	HiddenSize       uint32  `json:"hidden_size"`
	IntermediateSize uint32  `json:"intermediate_size"`
	MaxPositions     uint32  `json:"max_position_embeddings"`
	AttentionHeads   uint32  `json:"num_attention_heads"`
	HiddenLayers     uint32  `json:"num_hidden_layers"`
	KVHeads          uint32  `json:"num_key_value_heads"`
	HeadDim          uint32  `json:"head_dim"`
	RMSEpsilon       float32 `json:"rms_norm_eps"`
	RopeTheta        float32 `json:"rope_theta"`
	Vocabulary       uint32  `json:"vocab_size"`
	TiedEmbeddings   bool    `json:"tie_word_embeddings"`
}

type tokenizerFile struct {
	Added []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
	PostProcessor json.RawMessage `json:"post_processor"`
	Model         struct {
		UnknownToken *string           `json:"unk_token"`
		Vocab        map[string]int    `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
	} `json:"model"`
}

// Options: one HF-llama checkpoint export.
type Options struct {
	Directory  string
	OutputPath string
	Name       string
}

type Report struct {
	Tensors     int
	VocabSize   int
	OutputBytes int64
}

func Convert(options Options) (Report, error) {
	directory, err := filepath.Abs(options.Directory)
	if err != nil {
		return Report{}, err
	}
	if strings.TrimSpace(options.OutputPath) == "" {
		return Report{}, errors.New("HF-llama converter: output path is required")
	}
	if _, statErr := os.Stat(filepath.Join(directory, "model.safetensors.index.json")); statErr == nil {
		return Report{}, errors.New("HF-llama converter: sharded checkpoints are unsupported; need a single model.safetensors")
	}
	if _, statErr := os.Stat(filepath.Join(directory, "model.safetensors")); statErr != nil {
		return Report{}, fmt.Errorf("HF-llama converter: model.safetensors: %w", statErr)
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
	metadata, err := modelMetadata(directory, name, config)
	if err != nil {
		return Report{}, err
	}
	tensors, err := modelTensors(source, config)
	if err != nil {
		return Report{}, err
	}
	if err := validateModelCatalog(metadata, tensors); err != nil {
		return Report{}, err
	}
	if err := writeOutput(options.OutputPath, metadata, tensors); err != nil {
		return Report{}, err
	}
	written, err := os.Stat(options.OutputPath)
	if err != nil {
		return Report{}, err
	}
	return Report{
		Tensors: len(tensors), VocabSize: int(config.Vocabulary), OutputBytes: written.Size(),
	}, nil
}

func readConfig(directory string) (modelConfig, error) {
	var config modelConfig
	encoded, err := os.ReadFile(filepath.Join(directory, "config.json"))
	if err != nil {
		return config, fmt.Errorf("HF-llama converter: read config: %w", err)
	}
	if err := json.Unmarshal(encoded, &config); err != nil {
		return config, fmt.Errorf("HF-llama converter: parse config: %w", err)
	}
	return config, nil
}

func validateConfig(config modelConfig) error {
	if config.ModelType != "llama" {
		return fmt.Errorf("HF-llama converter: model type %q is unsupported; need llama", config.ModelType)
	}
	if config.HiddenLayers == 0 || config.HiddenSize == 0 || config.IntermediateSize == 0 ||
		config.AttentionHeads == 0 || config.KVHeads == 0 || config.Vocabulary == 0 ||
		config.MaxPositions == 0 || config.RMSEpsilon <= 0 || config.RopeTheta <= 0 {
		return errors.New("HF-llama converter: configuration is incomplete")
	}
	if config.HiddenSize%config.AttentionHeads != 0 {
		return errors.New("HF-llama converter: hidden size is not divisible by head count")
	}
	if config.HeadDim != 0 && config.HeadDim != config.HiddenSize/config.AttentionHeads {
		return errors.New("HF-llama converter: explicit head_dim disagrees with hidden_size/num_attention_heads")
	}
	return nil
}

func modelMetadata(directory, name string, config modelConfig) ([]gguf.Metadata, error) {
	metadata := []gguf.Metadata{
		stringMetadata("general.architecture", "llama"),
		stringMetadata("general.name", name),
		uint32Metadata("llama.block_count", config.HiddenLayers),
		uint32Metadata("llama.context_length", config.MaxPositions),
		uint32Metadata("llama.embedding_length", config.HiddenSize),
		uint32Metadata("llama.feed_forward_length", config.IntermediateSize),
		uint32Metadata("llama.attention.head_count", config.AttentionHeads),
		uint32Metadata("llama.attention.head_count_kv", config.KVHeads),
		float32Metadata("llama.attention.layer_norm_rms_epsilon", config.RMSEpsilon),
		float32Metadata("llama.rope.freq_base", config.RopeTheta),
		uint32Metadata("llama.rope.dimension_count", config.HiddenSize/config.AttentionHeads),
		uint32Metadata("llama.vocab_size", config.Vocabulary),
	}
	tokenizerItems, err := tokenizerMetadata(directory, config.Vocabulary)
	if err != nil {
		return nil, err
	}
	return append(metadata, tokenizerItems...), nil
}

func tokenizerMetadata(directory string, vocabulary uint32) ([]gguf.Metadata, error) {
	encoded, err := os.ReadFile(filepath.Join(directory, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("HF-llama converter: read tokenizer: %w", err)
	}
	var file tokenizerFile
	if err := json.Unmarshal(encoded, &file); err != nil {
		return nil, fmt.Errorf("HF-llama converter: parse tokenizer: %w", err)
	}
	pre, err := preTokenizerName(file.PreTokenizer)
	if err != nil {
		return nil, err
	}
	tokens := make([]string, vocabulary)
	types := make([]int32, vocabulary)
	seen := make([]bool, vocabulary)
	for token, id := range file.Model.Vocab {
		if id < 0 || id >= len(tokens) || seen[id] {
			return nil, fmt.Errorf("HF-llama converter: invalid tokenizer ID %d", id)
		}
		tokens[id], types[id], seen[id] = token, tokenTypeNormal, true
	}
	for _, added := range file.Added {
		if added.ID < 0 || added.ID >= len(tokens) {
			return nil, fmt.Errorf("HF-llama converter: added token ID %d is out of range", added.ID)
		}
		if seen[added.ID] && tokens[added.ID] != added.Content {
			return nil, fmt.Errorf("HF-llama converter: added token ID %d is inconsistent", added.ID)
		}
		tokens[added.ID], seen[added.ID] = added.Content, true
		types[added.ID] = tokenTypeUserDefined
		if added.Special {
			types[added.ID] = tokenTypeControl
		}
	}
	// llama.cpp pads embedding rows beyond the tokenizer with [PAD{id}] UNUSED tokens.
	for id, present := range seen {
		if !present {
			tokens[id], types[id] = "[PAD"+strconv.Itoa(id)+"]", tokenTypeUnused
		}
	}
	merges, err := mergeStrings(file.Model.Merges)
	if err != nil {
		return nil, err
	}
	special, err := resolveSpecialTokens(directory, file, tokens)
	if err != nil {
		return nil, err
	}
	if special.unknown >= 0 {
		types[special.unknown] = tokenTypeUnknown
	}
	metadata := []gguf.Metadata{
		stringMetadata("tokenizer.ggml.model", "gpt2"),
		stringMetadata("tokenizer.ggml.pre", pre),
		arrayMetadata("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		arrayMetadata("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
		arrayMetadata("tokenizer.ggml.merges", gguf.ValueTypeString, merges),
	}
	for _, item := range []struct {
		key string
		id  int
	}{
		{"tokenizer.ggml.bos_token_id", special.bos},
		{"tokenizer.ggml.eos_token_id", special.eos},
		{"tokenizer.ggml.unknown_token_id", special.unknown},
		{"tokenizer.ggml.padding_token_id", special.padding},
	} {
		if item.id >= 0 {
			metadata = append(metadata, uint32Metadata(item.key, uint32(item.id)))
		}
	}
	metadata = append(metadata,
		boolMetadata("tokenizer.ggml.add_bos_token", postProcessorAddsBOS(file.PostProcessor)),
		boolMetadata("tokenizer.ggml.add_eos_token", false),
	)
	return metadata, nil
}

const (
	tokenTypeNormal      = 1
	tokenTypeUnknown     = 2
	tokenTypeControl     = 3
	tokenTypeUserDefined = 4
	tokenTypeUnused      = 5
)

// HF pre-tokenizer Split regexes mapped to GGUF pre names the runtime supports.
const (
	gpt2SplitRegex  = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
	qwen2SplitRegex = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

var splitRegexPre = map[string]string{
	gpt2SplitRegex:  "gpt-2",
	qwen2SplitRegex: "qwen2",
}

type preTokenizerNode struct {
	Type    string `json:"type"`
	Pattern struct {
		Regex string `json:"Regex"`
	} `json:"pattern"`
	Pretokenizers []preTokenizerNode `json:"pretokenizers"`
}

func preTokenizerName(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "gpt-2", nil
	}
	var root preTokenizerNode
	if err := json.Unmarshal(raw, &root); err != nil {
		return "", fmt.Errorf("HF-llama converter: parse pre-tokenizer: %w", err)
	}
	var regexes []string
	collectSplitRegexes(root, &regexes)
	if len(regexes) == 0 {
		return "gpt-2", nil
	}
	if len(regexes) > 1 {
		return "", fmt.Errorf("HF-llama converter: %d pre-tokenizer split patterns; need at most one", len(regexes))
	}
	pre, ok := splitRegexPre[regexes[0]]
	if !ok {
		return "", fmt.Errorf("HF-llama converter: unsupported pre-tokenizer split regex %q", regexes[0])
	}
	return pre, nil
}

func collectSplitRegexes(node preTokenizerNode, out *[]string) {
	if node.Type == "Split" && node.Pattern.Regex != "" {
		*out = append(*out, node.Pattern.Regex)
	}
	for _, child := range node.Pretokenizers {
		collectSplitRegexes(child, out)
	}
}

type postProcessorNode struct {
	Type       string              `json:"type"`
	Single     []json.RawMessage   `json:"single"`
	Processors []postProcessorNode `json:"processors"`
}

// postProcessorAddsBOS: TemplateProcessing with a leading special token prepends BOS.
func postProcessorAddsBOS(raw json.RawMessage) bool {
	if len(raw) == 0 || string(raw) == "null" {
		return false
	}
	var root postProcessorNode
	if err := json.Unmarshal(raw, &root); err != nil {
		return false
	}
	return templateAddsSpecialPrefix(root)
}

func templateAddsSpecialPrefix(node postProcessorNode) bool {
	if node.Type == "TemplateProcessing" && len(node.Single) > 0 {
		var first struct {
			SpecialToken *struct{} `json:"SpecialToken"`
		}
		if json.Unmarshal(node.Single[0], &first) == nil && first.SpecialToken != nil {
			return true
		}
	}
	for _, child := range node.Processors {
		if templateAddsSpecialPrefix(child) {
			return true
		}
	}
	return false
}

func mergeStrings(raw []json.RawMessage) ([]string, error) {
	merges := make([]string, len(raw))
	for index, entry := range raw {
		var joined string
		if err := json.Unmarshal(entry, &joined); err == nil {
			if strings.Count(joined, " ") == 0 {
				return nil, fmt.Errorf("HF-llama converter: merge %d is invalid", index)
			}
			merges[index] = joined
			continue
		}
		var pair []string
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) != 2 || pair[0] == "" || pair[1] == "" {
			return nil, fmt.Errorf("HF-llama converter: merge %d is invalid", index)
		}
		merges[index] = pair[0] + " " + pair[1]
	}
	return merges, nil
}

type specialTokenIDs struct {
	bos, eos, unknown, padding int
}

type generationConfig struct {
	BOS json.RawMessage `json:"bos_token_id"`
	EOS json.RawMessage `json:"eos_token_id"`
	PAD json.RawMessage `json:"pad_token_id"`
}

type specialTokensMap struct {
	BOS json.RawMessage `json:"bos_token"`
	EOS json.RawMessage `json:"eos_token"`
	UNK json.RawMessage `json:"unk_token"`
	PAD json.RawMessage `json:"pad_token"`
}

// resolveSpecialTokens: generation_config IDs win, then special_tokens_map
// contents, then tokenizer.json model.unk_token; -1 means unresolved.
func resolveSpecialTokens(directory string, file tokenizerFile, tokens []string) (specialTokenIDs, error) {
	result := specialTokenIDs{bos: -1, eos: -1, unknown: -1, padding: -1}
	tokenID := make(map[string]int, len(tokens))
	for id, token := range tokens {
		tokenID[token] = id
	}
	var generation generationConfig
	if err := readOptionalJSON(filepath.Join(directory, "generation_config.json"), &generation); err != nil {
		return result, err
	}
	var special specialTokensMap
	if err := readOptionalJSON(filepath.Join(directory, "special_tokens_map.json"), &special); err != nil {
		return result, err
	}
	entries := []struct {
		destination *int
		id          json.RawMessage
		content     json.RawMessage
		fallback    *string
	}{
		{&result.bos, generation.BOS, special.BOS, nil},
		{&result.eos, generation.EOS, special.EOS, nil},
		{&result.unknown, nil, special.UNK, file.Model.UnknownToken},
		{&result.padding, generation.PAD, special.PAD, nil},
	}
	for _, entry := range entries {
		id, err := firstTokenID(entry.id, entry.content, entry.fallback, tokenID)
		if err != nil {
			return result, err
		}
		if id >= 0 && id < len(tokens) {
			*entry.destination = id
		} else if id >= len(tokens) {
			return result, fmt.Errorf("HF-llama converter: special token ID %d is out of range", id)
		}
	}
	return result, nil
}

func firstTokenID(rawID, rawContent json.RawMessage, fallback *string, tokenID map[string]int) (int, error) {
	if id, ok := decodeTokenID(rawID); ok {
		return id, nil
	}
	content, ok := decodeTokenContent(rawContent)
	if !ok && fallback != nil {
		content, ok = *fallback, true
	}
	if !ok {
		return -1, nil
	}
	id, found := tokenID[content]
	if !found {
		return -1, fmt.Errorf("HF-llama converter: special token %q is missing from the vocabulary", content)
	}
	return id, nil
}

// decodeTokenID: accepts an integer or the first element of an integer array.
func decodeTokenID(raw json.RawMessage) (int, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return -1, false
	}
	var id int
	if err := json.Unmarshal(raw, &id); err == nil {
		return id, true
	}
	var ids []int
	if err := json.Unmarshal(raw, &ids); err == nil && len(ids) > 0 {
		return ids[0], true
	}
	return -1, false
}

// decodeTokenContent: accepts a string or an added-token object with content.
func decodeTokenContent(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", false
	}
	var content string
	if err := json.Unmarshal(raw, &content); err == nil {
		return content, true
	}
	var object struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &object); err == nil && object.Content != "" {
		return object.Content, true
	}
	return "", false
}

func readOptionalJSON(path string, destination any) error {
	encoded, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("HF-llama converter: read %s: %w", filepath.Base(path), err)
	}
	if err := json.Unmarshal(encoded, destination); err != nil {
		return fmt.Errorf("HF-llama converter: parse %s: %w", filepath.Base(path), err)
	}
	return nil
}

var layerNamePattern = regexp.MustCompile(`^model\.layers\.(\d+)\.(.+)$`)

func modelTensors(source *safetensors.Source, config modelConfig) ([]gguf.TensorData, error) {
	tensors := make([]gguf.TensorData, 0, len(source.Tensors))
	sawOutput := false
	for sourceName, tensor := range source.Tensors {
		destinationName, ok := modelTensorName(sourceName)
		if !ok {
			return nil, fmt.Errorf("HF-llama converter: tensor %q has no mapping", sourceName)
		}
		if destinationName == "output.weight" {
			sawOutput = true
		}
		dataType, reader, err := modelTensorReader(tensor)
		if err != nil {
			return nil, fmt.Errorf("HF-llama converter: tensor %q: %w", sourceName, err)
		}
		// GGUF llama RoPE is interleaved; HF q/k rows are rotate-half layout
		if heads := ropePermuteHeads(destinationName, config); heads != 0 {
			reader, err = permuteRopeRows(reader, tensor.Shape, dataType, heads)
			if err != nil {
				return nil, fmt.Errorf("HF-llama converter: permute %q: %w", sourceName, err)
			}
		}
		tensors = append(tensors, gguf.TensorData{
			Name: destinationName, Shape: reverseShape(tensor.Shape), Type: dataType, Data: reader,
		})
	}
	// Tied embeddings: runtime output head falls back to token_embd; no duplicate tensor.
	if !sawOutput && !config.TiedEmbeddings {
		return nil, errors.New("HF-llama converter: lm_head.weight is missing and embeddings are untied")
	}
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })
	if len(tensors) == 0 {
		return nil, errors.New("HF-llama converter: no tensors found")
	}
	return tensors, nil
}

func modelTensorName(name string) (string, bool) {
	switch name {
	case "model.embed_tokens.weight":
		return "token_embd.weight", true
	case "model.norm.weight":
		return "output_norm.weight", true
	case "lm_head.weight":
		return "output.weight", true
	}
	match := layerNamePattern.FindStringSubmatch(name)
	if match == nil {
		return "", false
	}
	mapping := map[string]string{
		"self_attn.q_proj.weight":         "attn_q.weight",
		"self_attn.k_proj.weight":         "attn_k.weight",
		"self_attn.v_proj.weight":         "attn_v.weight",
		"self_attn.o_proj.weight":         "attn_output.weight",
		"mlp.gate_proj.weight":            "ffn_gate.weight",
		"mlp.up_proj.weight":              "ffn_up.weight",
		"mlp.down_proj.weight":            "ffn_down.weight",
		"input_layernorm.weight":          "attn_norm.weight",
		"post_attention_layernorm.weight": "ffn_norm.weight",
	}
	suffix, ok := mapping[match[2]]
	if !ok {
		return "", false
	}
	return "blk." + match[1] + "." + suffix, true
}

func ropePermuteHeads(destinationName string, config modelConfig) uint32 {
	switch {
	case strings.HasSuffix(destinationName, ".attn_q.weight"):
		return config.AttentionHeads
	case strings.HasSuffix(destinationName, ".attn_k.weight"):
		return config.KVHeads
	}
	return 0
}

// permuteRopeRows: llama.cpp q/k permutation; per head, row (s, i) of the
// (2, d/2) split moves to interleaved row 2i+s.
func permuteRopeRows(reader io.Reader, shape []uint64, dataType gguf.DType, heads uint32) (io.Reader, error) {
	if len(shape) != 2 || heads == 0 {
		return nil, errors.New("rank-2 tensor with head count required")
	}
	elementSize := uint64(0)
	switch dataType {
	case gguf.DTypeBF16, gguf.DTypeF16:
		elementSize = 2
	case gguf.DTypeF32:
		elementSize = 4
	default:
		return nil, fmt.Errorf("unsupported dtype %d", dataType)
	}
	outRows, rowBytes := shape[0], shape[1]*elementSize
	if outRows%uint64(heads) != 0 {
		return nil, errors.New("rows are not divisible by head count")
	}
	headDim := outRows / uint64(heads)
	if headDim%2 != 0 {
		return nil, errors.New("head dimension is odd")
	}
	source, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	if uint64(len(source)) != outRows*rowBytes {
		return nil, errors.New("tensor byte size disagrees with its shape")
	}
	destination := make([]byte, len(source))
	half := headDim / 2
	for head := uint64(0); head < uint64(heads); head++ {
		base := head * headDim
		for pair := uint64(0); pair < half; pair++ {
			for split := uint64(0); split < 2; split++ {
				sourceRow := base + split*half + pair
				destinationRow := base + pair*2 + split
				copy(
					destination[destinationRow*rowBytes:(destinationRow+1)*rowBytes],
					source[sourceRow*rowBytes:(sourceRow+1)*rowBytes],
				)
			}
		}
	}
	return bytes.NewReader(destination), nil
}

// modelTensorReader: source dtypes pass through unconverted.
func modelTensorReader(tensor safetensors.Tensor) (gguf.DType, io.Reader, error) {
	switch tensor.DType {
	case "BF16":
		return gguf.DTypeBF16, tensor.Reader(), nil
	case "F16":
		return gguf.DTypeF16, tensor.Reader(), nil
	case "F32":
		return gguf.DTypeF32, tensor.Reader(), nil
	default:
		return 0, nil, fmt.Errorf("unsupported dtype %q", tensor.DType)
	}
}

func validateModelCatalog(metadata []gguf.Metadata, tensors []gguf.TensorData) error {
	file := &gguf.File{Metadata: metadata, Tensors: make([]gguf.TensorInfo, len(tensors))}
	for index, tensor := range tensors {
		if len(tensor.Shape) > gguf.MaxDimensions {
			return fmt.Errorf("HF-llama converter: tensor %q rank exceeds GGUF", tensor.Name)
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
		return fmt.Errorf("HF-llama converter: generated model metadata: %w", err)
	}
	if _, err := model.ReadWeights(file, spec); err != nil {
		return fmt.Errorf("HF-llama converter: generated tensor catalog: %w", err)
	}
	if _, err := tokenizer.Load(file); err != nil {
		return fmt.Errorf("HF-llama converter: generated tokenizer: %w", err)
	}
	return nil
}

func writeOutput(path string, metadata []gguf.Metadata, tensors []gguf.TensorData) error {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(absolute, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return fmt.Errorf("HF-llama converter: create %s: %w", absolute, err)
	}
	succeeded := false
	defer func() {
		_ = file.Close()
		if !succeeded {
			_ = os.Remove(absolute)
		}
	}()
	if err := gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		return fmt.Errorf("HF-llama converter: write %s: %w", absolute, err)
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
