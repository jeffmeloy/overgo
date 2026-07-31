package inference

import (
	"fmt"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

// ModelProperties is the immutable model metadata exposed by llama.cpp's
// /props-compatible server endpoint. It intentionally contains only
// authoritative GGUF/runtime facts and no mutable generation state.
type ModelProperties struct {
	Path              string
	Name              string
	Architecture      string
	FileType          string
	ContextLength     uint32
	EmbeddingLength   uint32
	FeedForwardLength uint32
	BlockCount        uint32
	HeadCount         uint32
	HeadCountKV       uint32
	VocabularySize    uint32
	VocabularyType    string
	ParameterCount    uint64
	ModelSize         uint64
	ChatTemplate      string
	BOSToken          string
	EOSToken          string
}

// ModelProperties returns a detached snapshot of the loaded model's public
// metadata. It is safe to call without entering the generation mutex.
func (r *Runner) ModelProperties() ModelProperties {
	if r == nil {
		return ModelProperties{}
	}
	properties := ModelProperties{
		Path:              r.path,
		Name:              r.spec.Name,
		Architecture:      r.spec.Architecture,
		FileType:          metadataFileType(r.file),
		ContextLength:     r.spec.ContextLength,
		EmbeddingLength:   r.spec.EmbeddingLength,
		FeedForwardLength: r.spec.FeedForwardLength,
		BlockCount:        r.spec.BlockCount,
		HeadCount:         r.spec.HeadCount,
		HeadCountKV:       r.spec.HeadCountKV,
		VocabularySize:    r.spec.VocabularySize,
		ParameterCount:    modelParameterCount(r.file),
		ModelSize:         modelTensorBytes(r.file),
		ChatTemplate:      metadataString(r.file, "tokenizer.chat_template"),
	}
	if r.vocab != nil {
		properties.VocabularyType = r.vocab.Model
		properties.BOSToken = vocabularyTokenText(r.vocab, r.vocab.BOS)
		properties.EOSToken = vocabularyTokenText(r.vocab, r.vocab.EOS)
		if properties.VocabularySize == 0 {
			properties.VocabularySize = uint32(r.vocab.Len())
		}
	}
	return properties
}

func modelParameterCount(file *gguf.File) uint64 {
	if file == nil {
		return 0
	}
	var total uint64
	for _, tensor := range file.Tensors {
		count := uint64(1)
		for dimension := range tensor.Dimensions {
			size := tensor.Shape[dimension]
			if size == 0 || count > ^uint64(0)/size {
				return 0
			}
			count *= size
		}
		if total > ^uint64(0)-count {
			return 0
		}
		total += count
	}
	return total
}

func modelTensorBytes(file *gguf.File) uint64 {
	if file == nil {
		return 0
	}
	var total uint64
	for _, tensor := range file.Tensors {
		if total > ^uint64(0)-tensor.Size {
			return 0
		}
		total += tensor.Size
	}
	return total
}

func metadataString(file *gguf.File, key string) string {
	if file == nil {
		return ""
	}
	value, ok := modelMetadataValue(file, key)
	if !ok || value.Type != gguf.ValueTypeString {
		return ""
	}
	result, _ := value.Data.(string)
	return result
}

func vocabularyTokenText(vocab *tokenizer.Vocab, id tokenizer.TokenID) string {
	token, ok := vocab.Token(id)
	if !ok {
		return ""
	}
	return token.Text
}

func metadataFileType(file *gguf.File) string {
	if file == nil {
		return ""
	}
	value, ok := modelMetadataValue(file, "general.file_type")
	if !ok || value.Type != gguf.ValueTypeUint32 {
		return ""
	}
	fileType, ok := value.Data.(uint32)
	if !ok {
		return ""
	}
	if name, ok := ggufFileTypeNames[fileType]; ok {
		return name
	}
	return fmt.Sprintf("unknown (%d)", fileType)
}

func modelMetadataValue(file *gguf.File, key string) (gguf.Value, bool) {
	if file == nil {
		return gguf.Value{}, false
	}
	if value, ok := file.MetadataValue(key); ok {
		return value, true
	}
	// Parsed GGUF files have an index; the linear fallback also makes detached
	// metadata snapshots and unit fixtures behave like parsed files.
	for _, metadata := range file.Metadata {
		if metadata.Key == key {
			return metadata.Value, true
		}
	}
	return gguf.Value{}, false
}

// These names follow llama_ftype_name() at the pinned llama.cpp commit.
var ggufFileTypeNames = map[uint32]string{
	0:  "all F32",
	1:  "F16",
	2:  "Q4_0",
	3:  "Q4_1",
	7:  "Q8_0",
	8:  "Q5_0",
	9:  "Q5_1",
	10: "Q2_K - Medium",
	11: "Q3_K - Small",
	12: "Q3_K - Medium",
	13: "Q3_K - Large",
	14: "Q4_K - Small",
	15: "Q4_K - Medium",
	16: "Q5_K - Small",
	17: "Q5_K - Medium",
	18: "Q6_K",
	19: "IQ2_XXS - 2.0625 bpw",
	20: "IQ2_XS - 2.3125 bpw",
	21: "Q2_K - Small",
	22: "IQ3_XS - 3.3 bpw",
	23: "IQ3_XXS - 3.0625 bpw",
	24: "IQ1_S - 1.5625 bpw",
	25: "IQ4_NL - 4.5 bpw",
	26: "IQ3_S - 3.4375 bpw",
	27: "IQ3_S mix - 3.66 bpw",
	28: "IQ2_S - 2.5 bpw",
	29: "IQ2_M - 2.7 bpw",
	30: "IQ4_XS - 4.25 bpw",
	31: "IQ1_M - 1.75 bpw",
	32: "BF16",
	36: "TQ1_0 - 1.69 bpw ternary",
	37: "TQ2_0 - 2.06 bpw ternary",
	38: "MXFP4 MoE",
	39: "NVFP4",
	40: "Q1_0",
	41: "Q2_0",
}
