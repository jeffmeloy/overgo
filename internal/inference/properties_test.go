package inference

import (
	"os"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tokenizer"
)

func TestModelPropertiesReturnsDetachedGGUFMetadata(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{path: "model.gguf",
		file: &gguf.File{
			Metadata: []gguf.Metadata{{
				Key: "general.file_type",
				Value: gguf.Value{
					Type: gguf.ValueTypeUint32,
					Data: uint32(30),
				},
			},
				{
					Key: "tokenizer.chat_template",
					Value: gguf.Value{
						Type: gguf.ValueTypeString,
						Data: "{{ messages }}",
					},
				}},
			Tensors: []gguf.TensorInfo{
				{Dimensions: 2, Shape: [gguf.MaxDimensions]uint64{2, 3}, Size: 12},
				{Dimensions: 1, Shape: [gguf.MaxDimensions]uint64{4}, Size: 8},
			},
		},
		spec: model.Spec{CommonSpec: model.CommonSpec{Name: "fixture",
			Architecture:      "qwen3",
			ContextLength:     32768,
			EmbeddingLength:   2560,
			FeedForwardLength: 9728,
			BlockCount:        36,

			VocabularySize: 3}, AttentionSpec: model.AttentionSpec{HeadCount: 32,
			HeadCountKV: 8},
		},
		vocab: &tokenizer.Vocab{
			Model: "gpt2",
			Tokens: []tokenizer.Token{
				{Text: "normal"},
				{Text: "<bos>"},
				{Text: "<eos>"},
			},
			BOS: 1,
			EOS: 2,
		}},
	}

	properties := runner.ModelProperties()
	if properties.Path != "model.gguf" ||
		properties.Name != "fixture" ||
		properties.Architecture != "qwen3" ||
		properties.FileType != "IQ4_XS - 4.25 bpw" ||
		properties.ContextLength != 32768 ||
		properties.VocabularySize != 3 ||
		properties.VocabularyType != "gpt2" ||
		properties.ParameterCount != 10 ||
		properties.ModelSize != 20 ||
		properties.ChatTemplate != "{{ messages }}" ||
		properties.BOSToken != "<bos>" ||
		properties.EOSToken != "<eos>" {
		t.Fatalf("properties = %+v", properties)
	}
}

func TestModelPropertiesFromRealGGUF(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_QWEN3_MODEL")
	if path == "" {
		t.Skip("LLAMACPP2GO_QWEN3_MODEL is not set")
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	vocab, err := tokenizer.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	runner := &Runner{preparedModel: preparedModel{path: path,
		file:  file,
		spec:  spec,
		vocab: vocab},
	}
	properties := runner.ModelProperties()
	if properties.Path != path ||
		properties.Name == "" ||
		properties.Architecture != "qwen3" ||
		properties.FileType == "" ||
		properties.ContextLength == 0 ||
		properties.EmbeddingLength == 0 ||
		properties.BlockCount == 0 ||
		properties.VocabularySize != uint32(vocab.Len()) ||
		properties.EOSToken == "" {
		t.Fatalf("properties = %+v", properties)
	}
	if properties.VocabularyType == "" ||
		properties.ParameterCount == 0 ||
		properties.ModelSize == 0 {
		t.Fatalf("incomplete model counts = %+v", properties)
	}
}

func TestModelPropertiesHandlesUnavailableOptionalMetadata(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{file: &gguf.File{},
		spec: model.Spec{CommonSpec: model.CommonSpec{VocabularySize: 0}},
		vocab: &tokenizer.Vocab{
			Tokens: []tokenizer.Token{{Text: "x"}},
			BOS:    tokenizer.NullToken,
			EOS:    tokenizer.NullToken,
		}},
	}
	properties := runner.ModelProperties()
	if properties.FileType != "" ||
		properties.ChatTemplate != "" ||
		properties.BOSToken != "" ||
		properties.EOSToken != "" ||
		properties.VocabularySize != 1 {
		t.Fatalf("properties = %+v", properties)
	}
}

func TestModelPropertyCountsRejectOverflow(t *testing.T) {
	file := &gguf.File{Tensors: []gguf.TensorInfo{{
		Dimensions: 2,
		Shape:      [gguf.MaxDimensions]uint64{^uint64(0), 2},
		Size:       ^uint64(0),
	}, {
		Dimensions: 1,
		Shape:      [gguf.MaxDimensions]uint64{1},
		Size:       1,
	}}}
	if modelParameterCount(file) != 0 {
		t.Fatal("overflowing parameter count was accepted")
	}
	if modelTensorBytes(file) != 0 {
		t.Fatal("overflowing tensor byte count was accepted")
	}
}
