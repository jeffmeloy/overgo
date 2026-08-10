package server

import (
	"net/http"

	"overgo/internal/inference"
)

// analyzeModelResponse: read-only aggregate of the loaded model's architecture
// and measured statistics for the GUI analysis workbench. Every value is an
// exact fact drawn from the GGUF/runtime metadata or an integer ratio of such
// facts — nothing here fits a distribution or assumes a shape (see the
// workbench's distribution-free analysis principle).
type analyzeModelResponse struct {
	Model        analyzeModelFacts   `json:"model"`
	Derived      analyzeModelDerived `json:"derived"`
	Capabilities []string            `json:"capabilities"`
}

type analyzeModelFacts struct {
	ID                string `json:"id"`
	Name              string `json:"name"`
	Architecture      string `json:"architecture"`
	FileType          string `json:"file_type"`
	Path              string `json:"path"`
	Parameters        uint64 `json:"parameters"`
	SizeBytes         uint64 `json:"size_bytes"`
	ContextLength     uint32 `json:"context_length"`
	EmbeddingLength   uint32 `json:"embedding_length"`
	FeedForwardLength uint32 `json:"feed_forward_length"`
	BlockCount        uint32 `json:"block_count"`
	HeadCount         uint32 `json:"head_count"`
	HeadCountKV       uint32 `json:"head_count_kv"`
	VocabularySize    uint32 `json:"vocabulary_size"`
	VocabularyType    string `json:"vocabulary_type"`
	BOSToken          string `json:"bos_token"`
	EOSToken          string `json:"eos_token"`
	ChatTemplate      string `json:"chat_template"`
}

// analyzeModelDerived: exact ratios of the facts above. Pointer fields are
// omitted when their divisor is zero rather than reported as a guessed value.
type analyzeModelDerived struct {
	HeadDim          *uint64  `json:"head_dim,omitempty"`
	KVGroupSize      *uint64  `json:"kv_group_size,omitempty"`
	GroupedQuery     bool     `json:"grouped_query"`
	ParamsPerBlock   *uint64  `json:"params_per_block,omitempty"`
	AvgBitsPerWeight *float64 `json:"avg_bits_per_weight,omitempty"`
	ChatTemplateSize int      `json:"chat_template_bytes"`
}

func (h *Handler) analyzeModel(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	api, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "model properties are unavailable")
		return
	}
	model := api.ModelProperties()
	result := analyzeModelResponse{
		Model: analyzeModelFacts{
			ID:                h.config.ModelID,
			Name:              model.Name,
			Architecture:      model.Architecture,
			FileType:          model.FileType,
			Path:              model.Path,
			Parameters:        model.ParameterCount,
			SizeBytes:         model.ModelSize,
			ContextLength:     model.ContextLength,
			EmbeddingLength:   model.EmbeddingLength,
			FeedForwardLength: model.FeedForwardLength,
			BlockCount:        model.BlockCount,
			HeadCount:         model.HeadCount,
			HeadCountKV:       model.HeadCountKV,
			VocabularySize:    model.VocabularySize,
			VocabularyType:    model.VocabularyType,
			BOSToken:          model.BOSToken,
			EOSToken:          model.EOSToken,
			ChatTemplate:      model.ChatTemplate,
		},
		Derived:      deriveModelStatistics(model),
		Capabilities: h.modelCapabilities(),
	}
	writeJSON(response, http.StatusOK, result)
}

// deriveModelStatistics: integer/exact ratios only; a divisor of zero leaves
// the corresponding field absent instead of fabricating a value.
func deriveModelStatistics(model inference.ModelProperties) analyzeModelDerived {
	derived := analyzeModelDerived{
		GroupedQuery:     model.HeadCountKV != 0 && model.HeadCount != model.HeadCountKV,
		ChatTemplateSize: len(model.ChatTemplate),
	}
	if model.HeadCount != 0 {
		headDim := uint64(model.EmbeddingLength) / uint64(model.HeadCount)
		derived.HeadDim = &headDim
	}
	if model.HeadCountKV != 0 {
		group := uint64(model.HeadCount) / uint64(model.HeadCountKV)
		derived.KVGroupSize = &group
	}
	if model.BlockCount != 0 {
		perBlock := model.ParameterCount / uint64(model.BlockCount)
		derived.ParamsPerBlock = &perBlock
	}
	if model.ParameterCount != 0 {
		bits := 8 * float64(model.ModelSize) / float64(model.ParameterCount)
		derived.AvgBitsPerWeight = &bits
	}
	return derived
}

// modelCapabilities: the same capability probe the /models endpoint reports,
// factored out so the analysis surface stays in lockstep with it.
func (h *Handler) modelCapabilities() []string {
	capabilities := []string{"completion", "embedding"}
	if capability, ok := h.generator.(RankCapability); ok && capability.SupportsRank() {
		capabilities = append(capabilities, "rerank")
	}
	return capabilities
}
