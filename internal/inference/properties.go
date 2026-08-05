package inference

import (
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

// ModelProperties: immutable model metadata exposed by llama.cpp's
// /props-compatible server endpoint; contains only
// authoritative GGUF/runtime facts and no mutable generation state
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

// ModelProperties: returns detached snapshot of loaded model's public
// metadata; safe to call without entering generation mutex
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
	return gguf.FileTypeName(fileType)
}

func modelMetadataValue(file *gguf.File, key string) (gguf.Value, bool) {
	if file == nil {
		return gguf.Value{}, false
	}
	if value, ok := file.MetadataValue(key); ok {
		return value, true
	}
	// Parsed GGUF files have index; linear fallback also makes detached
	// metadata snapshots and unit fixtures behave like parsed files
	for _, metadata := range file.Metadata {
		if metadata.Key == key {
			return metadata.Value, true
		}
	}
	return gguf.Value{}, false
}
