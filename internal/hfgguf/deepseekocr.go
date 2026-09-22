package hfgguf

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

// The DeepSeek-OCR family: a DeepseekV2 mixture-of-experts decoder under a
// document encoder of a SAM tower, a CLIP tower and a linear projector. The
// checkpoint declares the decoder under language_config with its own
// architecture; the wrapper's model type names the product, which this
// converter never reads. The language conversion writes the decoder for the
// deepseek2-ocr profile: dense feed-forward for the leading blocks, then a
// router, stacked routed experts and shared experts per block.
const (
	deepSeekOCRArchitecture         = "deepseek2-ocr"
	deepSeekOCRLanguageArchitecture = "DeepseekOCRForCausalLM"
	deepSeekOCRLanguageConfigKey    = "language_config"
	deepSeekOCRExpertPrefix         = "mlp.experts."
)

// deepSeekOCRLayerNames extends the dense layer table with the router and
// the shared experts; the routed experts are stacked, not mapped one by one.
var deepSeekOCRLayerNames = func() map[string]string {
	names := make(map[string]string, len(denseLayerNames)+4)
	maps.Copy(names, denseLayerNames)
	names["mlp.gate.weight"] = "ffn_gate_inp.weight"
	names["mlp.shared_experts.gate_proj.weight"] = "ffn_gate_shexp.weight"
	names["mlp.shared_experts.up_proj.weight"] = "ffn_up_shexp.weight"
	names["mlp.shared_experts.down_proj.weight"] = "ffn_down_shexp.weight"
	return names
}()

// tensor2D is the rank of a projection matrix: rows and columns.
const tensor2D = tensor.PairedExtent

// deepSeekOCRExpertProjections are the routed expert projections in the
// order each block stacks them.
var deepSeekOCRExpertProjections = []struct{ hf, gguf string }{
	{"gate_proj.weight", "ffn_gate_exps.weight"},
	{"up_proj.weight", "ffn_up_exps.weight"},
	{"down_proj.weight", "ffn_down_exps.weight"},
}

// IsDeepSeekOCRRepository reports a checkpoint whose decoder declares the
// DeepSeek-OCR language architecture.
func IsDeepSeekOCRRepository(identity hfrepo.Identity) bool {
	return slices.Contains(identity.TextArchitectures, deepSeekOCRLanguageArchitecture)
}

// deepSeekOCRLayout is the decoder's shape as the language config declares it.
type deepSeekOCRLayout struct {
	blocks, leadingDense, experts, expertsUsed, sharedExperts uint32
	expertFeedForward                                         uint32
}

// ValidateDeepSeekOCRRepository adapts the decoder and proves the loader
// reads it: the spec and the weight catalog resolve against the profile
// before any byte is written.
func ValidateDeepSeekOCRRepository(repository *hfrepo.Repository) (model.Spec, error) {
	metadata, layout, err := deepSeekOCRMetadata(repository)
	if err != nil {
		return model.Spec{}, err
	}
	mappings, stacks, err := deepSeekOCRTensorMappings(repository, layout)
	if err != nil {
		return model.Spec{}, err
	}
	infos, err := mappedTensorCatalog(mappings)
	if err != nil {
		return model.Spec{}, err
	}
	for _, stack := range stacks {
		infos = append(infos, stack.info())
	}
	return validateCatalog(metadata, infos, "DeepSeek-OCR")
}

// DeepSeekOCRConversion returns decoder metadata and streamed language
// tensors for GGUF export; tokenizer metadata is the converter's own.
func DeepSeekOCRConversion(repository *hfrepo.Repository) ([]gguf.Metadata, []gguf.TensorData, error) {
	metadata, layout, err := deepSeekOCRMetadata(repository)
	if err != nil {
		return nil, nil, err
	}
	mappings, stacks, err := deepSeekOCRTensorMappings(repository, layout)
	if err != nil {
		return nil, nil, err
	}
	tensors, err := mappedTensorData(mappings)
	if err != nil {
		return nil, nil, err
	}
	for _, stack := range stacks {
		tensors = append(tensors, stack.data())
	}
	return metadata, tensors, nil
}

// DeepSeekOCRLanguageConfig returns the decoder's configuration block.
func DeepSeekOCRLanguageConfig(repository *hfrepo.Repository) (map[string]json.RawMessage, error) {
	if repository == nil {
		return nil, errors.New("HF/GGUF adapter: nil repository")
	}
	if !IsDeepSeekOCRRepository(repository.Identity) {
		return nil, fmt.Errorf("HF/GGUF adapter: decoder architectures %v are not %s", repository.Identity.TextArchitectures, deepSeekOCRLanguageArchitecture)
	}
	var language map[string]json.RawMessage
	if err := json.Unmarshal(repository.Config[deepSeekOCRLanguageConfigKey], &language); err != nil || len(language) == 0 {
		return nil, errors.New("HF/GGUF adapter: DeepSeek-OCR checkpoint declares no language config")
	}
	return language, nil
}

func deepSeekOCRMetadata(repository *hfrepo.Repository) ([]gguf.Metadata, deepSeekOCRLayout, error) {
	language, err := DeepSeekOCRLanguageConfig(repository)
	if err != nil {
		return nil, deepSeekOCRLayout{}, err
	}
	dimensions, err := requiredValues[uint32](language,
		"max_position_embeddings", "hidden_size", "num_hidden_layers", "intermediate_size",
		"num_attention_heads", "num_key_value_heads", "vocab_size", "v_head_dim",
		"n_routed_experts", "num_experts_per_tok", "n_shared_experts", "moe_intermediate_size", "first_k_dense_replace",
	)
	if err != nil {
		return nil, deepSeekOCRLayout{}, err
	}
	contextLength, embeddingLength, blockCount, feedForwardLength := dimensions[0], dimensions[1], dimensions[2], dimensions[3]
	headCount, kvHeadCount, vocabulary, headLength := dimensions[4], dimensions[5], dimensions[6], dimensions[7]
	layout := deepSeekOCRLayout{
		blocks: blockCount, experts: dimensions[8], expertsUsed: dimensions[9], sharedExperts: dimensions[10],
		expertFeedForward: dimensions[11], leadingDense: dimensions[12],
	}
	if headCount == 0 || embeddingLength == 0 || headLength == 0 || embeddingLength != headCount*headLength {
		return nil, deepSeekOCRLayout{}, errors.New("HF/GGUF adapter: DeepSeek-OCR attention dimensions disagree with the hidden size")
	}
	if layout.experts == 0 || layout.expertsUsed == 0 || layout.expertsUsed > layout.experts || layout.expertFeedForward == 0 || layout.leadingDense >= blockCount {
		return nil, deepSeekOCRLayout{}, errors.New("HF/GGUF adapter: DeepSeek-OCR expert layout is invalid")
	}
	// The decoder's latent-attention ranks are null: it attends with plain
	// heads, and the profile carries no latent layout.
	for _, key := range []string{"use_mla"} {
		if value, ok, err := optional[bool](language, key); err != nil {
			return nil, deepSeekOCRLayout{}, err
		} else if ok && value {
			return nil, deepSeekOCRLayout{}, errors.New("HF/GGUF adapter: DeepSeek-OCR with latent attention is not the profile this converter writes")
		}
	}
	// A value the decoder leaves implicit takes the DeepseekV2 default.
	ropeBase, _, err := optional[float32](language, "rope_theta")
	if err != nil {
		return nil, deepSeekOCRLayout{}, err
	}
	normEpsilon, _, err := optional[float32](language, "rms_norm_eps")
	if err != nil {
		return nil, deepSeekOCRLayout{}, err
	}
	weightsScale, _, err := optional[float32](language, "routed_scaling_factor")
	if err != nil {
		return nil, deepSeekOCRLayout{}, err
	}
	ropeBase = cmp.Or(ropeBase, deepSeekOCRDefaultRopeBase)
	normEpsilon = cmp.Or(normEpsilon, deepSeekOCRDefaultNormEpsilon)
	weightsScale = cmp.Or(weightsScale, deepSeekOCRDefaultWeightsScale)
	prefix := deepSeekOCRArchitecture + "."
	return []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, deepSeekOCRArchitecture),
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
		metadata(prefix+"rope.dimension_count", gguf.ValueTypeUint32, headLength),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, normEpsilon),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, vocabulary),
		metadata(prefix+"expert_count", gguf.ValueTypeUint32, layout.experts),
		metadata(prefix+"expert_used_count", gguf.ValueTypeUint32, layout.expertsUsed),
		metadata(prefix+"expert_shared_count", gguf.ValueTypeUint32, layout.sharedExperts),
		metadata(prefix+"expert_feed_forward_length", gguf.ValueTypeUint32, layout.expertFeedForward),
		metadata(prefix+"leading_dense_block_count", gguf.ValueTypeUint32, layout.leadingDense),
		metadata(prefix+"expert_weights_scale", gguf.ValueTypeFloat32, weightsScale),
	}, layout, nil
}

// DeepseekV2 configuration defaults the checkpoint may leave implicit.
const (
	deepSeekOCRDefaultRopeBase     = float32(10000)
	deepSeekOCRDefaultNormEpsilon  = float32(1e-6)
	deepSeekOCRDefaultWeightsScale = float32(1)
)

// deepSeekOCRTensorMappings maps every decoder tensor the loader reads: the
// per-tensor mappings, and one stack per block and projection for the routed
// experts, whose payloads are the experts' rows in order.
func deepSeekOCRTensorMappings(repository *hfrepo.Repository, layout deepSeekOCRLayout) ([]tensorMapping, []expertStack, error) {
	if repository == nil || repository.Tensors == nil {
		return nil, nil, errors.New("HF/GGUF adapter: nil repository")
	}
	mappings, err := collectTensorMappings(repository.Tensors, deepSeekOCRTensorName, nil, nil)
	if err != nil {
		return nil, nil, err
	}
	var stacks []expertStack
	for block := layout.leadingDense; block < layout.blocks; block++ {
		for _, projection := range deepSeekOCRExpertProjections {
			stack, err := stackExperts(repository.Tensors, block, layout.experts, projection.hf, projection.gguf)
			if err != nil {
				return nil, nil, err
			}
			stacks = append(stacks, stack)
		}
	}
	return mappings, stacks, nil
}

// deepSeekOCRTensorName maps a decoder tensor to its GGUF name, leaving the
// encoder, the projector and the routed experts to their own conversions.
func deepSeekOCRTensorName(name string) (string, bool, error) {
	switch name {
	case "model.embed_tokens.weight":
		return "token_embd.weight", true, nil
	case "model.norm.weight":
		return "output_norm.weight", true, nil
	case "lm_head.weight":
		return "output.weight", true, nil
	}
	if !strings.HasPrefix(name, "model.layers.") {
		return "", false, nil // the encoder, the projector and their separators
	}
	rest, _ := strings.CutPrefix(name, "model.layers.")
	if _, suffix, ok := strings.Cut(rest, "."); ok && strings.HasPrefix(suffix, deepSeekOCRExpertPrefix) {
		return "", false, nil // stacked per block
	}
	if strings.HasSuffix(name, ".self_attn.rotary_emb.inv_freq") {
		return "", false, nil
	}
	mapped, _, err := mapLayerTensor(name, "model.layers.", 0, deepSeekOCRLayerNames)
	if err != nil {
		return "", false, fmt.Errorf("HF/GGUF adapter: tensor %q has no DeepSeek-OCR mapping", name)
	}
	return mapped, true, nil
}

// expertStack is one block's routed expert projection: the experts' matrices
// concatenated along a leading expert axis, streamed in expert order.
type expertStack struct {
	name    string
	experts []safetensors.Tensor
	shape   []uint64 // numpy order: experts, rows, columns
	storage gguf.DType
}

func stackExperts(source *safetensors.Source, block, experts uint32, hfSuffix, ggufSuffix string) (expertStack, error) {
	stack := expertStack{name: fmt.Sprintf("blk.%d.%s", block, ggufSuffix)}
	for expert := range experts {
		name := fmt.Sprintf("model.layers.%d.%s%d.%s", block, deepSeekOCRExpertPrefix, expert, hfSuffix)
		tensor, ok := source.Tensors[name]
		if !ok {
			return expertStack{}, fmt.Errorf("HF/GGUF adapter: routed expert tensor %q is absent", name)
		}
		if len(tensor.Shape) != tensor2D {
			return expertStack{}, fmt.Errorf("HF/GGUF adapter: routed expert tensor %q is not a matrix", name)
		}
		if expert == 0 {
			storage, ok := ggufStorage(tensor.DType, false)
			if !ok {
				return expertStack{}, fmt.Errorf("HF/GGUF adapter: tensor %q dtype %q needs conversion", name, tensor.DType)
			}
			stack.storage = storage
			stack.shape = []uint64{uint64(experts), tensor.Shape[0], tensor.Shape[1]}
		} else if tensor.Shape[0] != stack.shape[1] || tensor.Shape[1] != stack.shape[2] || ggufStorageName(tensor.DType) != ggufStorageName(stack.experts[0].DType) {
			return expertStack{}, fmt.Errorf("HF/GGUF adapter: routed expert tensor %q disagrees with expert 0", name)
		}
		stack.experts = append(stack.experts, tensor)
	}
	return stack, nil
}

func ggufStorageName(dataType string) string { return strings.ToUpper(strings.TrimSpace(dataType)) }

func (stack expertStack) info() gguf.TensorInfo {
	info := gguf.TensorInfo{Name: stack.name, Type: stack.storage, Dimensions: uint32(len(stack.shape))}
	for index, dimension := range stack.shape {
		info.Shape[len(stack.shape)-1-index] = dimension
	}
	traits, _ := stack.storage.Traits()
	elements := uint64(1)
	for _, dimension := range stack.shape {
		if dimension != 0 && elements > math.MaxUint64/dimension {
			return info
		}
		elements *= dimension
	}
	info.Size = elements * traits.TypeSize
	return info
}

func (stack expertStack) data() gguf.TensorData {
	readers := make([]io.Reader, len(stack.experts))
	for index, tensor := range stack.experts {
		readers[index] = tensor.Reader()
	}
	return gguf.TensorData{Name: stack.name, Shape: gguf.ReverseShape(stack.shape), Type: stack.storage, Data: io.MultiReader(readers...)}
}

// validateCatalog proves the loader reads a metadata and tensor catalog: the
// spec resolves and every weight the profile requires is present with its
// shape.
func validateCatalog(metadata []gguf.Metadata, tensors []gguf.TensorInfo, label string) (model.Spec, error) {
	file := &gguf.File{Metadata: metadata, Tensors: tensors}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return model.Spec{}, fmt.Errorf("HF/GGUF adapter: %s model spec: %w", label, err)
	}
	if _, err := model.ReadWeights(file, spec); err != nil {
		return model.Spec{}, fmt.Errorf("HF/GGUF adapter: %s weight catalog: %w", label, err)
	}
	return spec, nil
}
