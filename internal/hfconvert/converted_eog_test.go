package hfconvert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestConvertedEOGTokenList(t *testing.T) {
	t.Run("token_content_fallback", func(t *testing.T) {
		for _, declaration := range []struct {
			name, model, generation string
			want                    []tokenizer.TokenID
		}{
			{"absent", "", "", []tokenizer.TokenID{2}},
			{"null", "null", "null", []tokenizer.TokenID{2}},
			{"empty", "1", "[]", []tokenizer.TokenID{2}},
			{"model_precedence", "1", "null", []tokenizer.TokenID{1}},
			{"generation_precedence", "1", "[3,1]", []tokenizer.TokenID{1, 3}},
		} {
			t.Run(declaration.name, func(t *testing.T) {
				directory := t.TempDir()
				testutil.WriteTextFile(t, directory, "tokenizer.json", `{"model":{"vocab":{"a":0,"stop-a":1,"stop-b":2,"stop-c":3},"merges":[]}}`)
				testutil.WriteTextFile(t, directory, "special_tokens_map.json", `{"eos_token":{"content":"stop-b"}}`)
				if declaration.generation != "" {
					testutil.WriteTextFile(t, directory, "generation_config.json", `{"eos_token_id":`+declaration.generation+`}`)
				}
				metadata, err := tokenizerMetadata(directory, 4, json.RawMessage(declaration.model))
				if err != nil {
					t.Fatal(err)
				}
				vocab, err := tokenizer.Load(&gguf.File{Metadata: metadata})
				if err != nil {
					t.Fatal(err)
				}
				if actual := vocab.EOGTokens(); !slices.Equal(actual, declaration.want) {
					t.Fatalf("token-content fallback stops=%v, want %v", actual, declaration.want)
				}
			})
		}
	})
	t.Run("nested_qwen_config", func(t *testing.T) {
		for _, declaration := range []struct {
			name, source, generation string
			want                     []tokenizer.TokenID
		}{
			{"scalar", `{"text_config":{"vocab_size":4,"eos_token_id":1}}`, "", []tokenizer.TokenID{1}},
			{"array", `{"text_config":{"vocab_size":4,"eos_token_id":[1,2]}}`, "", []tokenizer.TokenID{1, 2}},
			{"generation_override", `{"text_config":{"vocab_size":4,"eos_token_id":[1,2]}}`, `{"eos_token_id":3}`, []tokenizer.TokenID{3}},
		} {
			t.Run(declaration.name, func(t *testing.T) {
				directory := t.TempDir()
				testutil.WriteTextFile(t, directory, "tokenizer.json", `{"model":{"vocab":{"a":0,"stop-a":1,"stop-b":2,"stop-c":3},"merges":[]}}`)
				if declaration.generation != "" {
					testutil.WriteTextFile(t, directory, "generation_config.json", declaration.generation)
				}
				var root map[string]json.RawMessage
				if err := json.Unmarshal([]byte(declaration.source), &root); err != nil {
					t.Fatal(err)
				}
				var text qwen35TokenConfig
				if err := json.Unmarshal(root["text_config"], &text); err != nil {
					t.Fatal(err)
				}
				metadata, err := tokenizerMetadata(directory, text.Vocabulary, text.EOS)
				if err != nil {
					t.Fatal(err)
				}
				vocab, err := tokenizer.Load(&gguf.File{Metadata: metadata})
				if err != nil {
					t.Fatal(err)
				}
				if actual := vocab.EOGTokens(); !slices.Equal(actual, declaration.want) {
					t.Fatalf("nested Qwen stops=%v, want %v", actual, declaration.want)
				}
			})
		}
	})
	for _, test := range []struct {
		name, config, generation string
		invalid                  bool
		want                     []tokenizer.TokenID
		primary                  tokenizer.TokenID
	}{
		{"generation_array", `{}`, `{"eos_token_id":[1,2,3]}`, false, []tokenizer.TokenID{1, 2, 3}, 1},
		{"more_than_eight", `{}`, `{"eos_token_id":[1,2,3,5,6,7,8,9,10,11]}`, false, []tokenizer.TokenID{1, 2, 3, 5, 6, 7, 8, 9, 10, 11}, 1},
		{"config_only", `{"eos_token_id":[1,2,3]}`, "", false, []tokenizer.TokenID{1, 2, 3}, 1},
		{"config_scalar", `{"eos_token_id":1}`, "", false, []tokenizer.TokenID{1}, 1},
		{"generation_precedence", `{"eos_token_id":[1,2]}`, `{"eos_token_id":3}`, false, []tokenizer.TokenID{3}, 3},
		{"generation_zero", `{"eos_token_id":1}`, `{"eos_token_id":0}`, false, []tokenizer.TokenID{0}, 0},
		{"generation_null_fallback", `{"eos_token_id":1}`, `{"eos_token_id":null}`, false, []tokenizer.TokenID{1}, 1},
		{"generation_unspecified_fallback", `{"eos_token_id":1}`, `{}`, false, []tokenizer.TokenID{1}, 1},
		{"no_declared_eos", `{}`, "", false, nil, tokenizer.NullToken},
		{"generation_empty", `{"eos_token_id":1}`, `{"eos_token_id":[]}`, false, nil, tokenizer.NullToken},
		{"invalid_later_id", `{}`, `{"eos_token_id":[1,99]}`, true, nil, tokenizer.NullToken},
		{"null_later_id", `{}`, `{"eos_token_id":[1,null]}`, true, nil, tokenizer.NullToken},
		{"negative_later_id", `{}`, `{"eos_token_id":[1,-1]}`, true, nil, tokenizer.NullToken},
		{"fractional_later_id", `{}`, `{"eos_token_id":[1,2.5]}`, true, nil, tokenizer.NullToken},
		{"duplicate_ids", `{}`, `{"eos_token_id":[1,2,1,3,2]}`, false, []tokenizer.TokenID{1, 2, 3}, 1},
		{"unordered_ids", `{}`, `{"eos_token_id":[3,1,3]}`, false, []tokenizer.TokenID{1, 3}, 3},
	} {
		t.Run(test.name, func(t *testing.T) {
			directory := t.TempDir()
			for name, content := range map[string]string{
				"tokenizer.json":         `{"model":{"vocab":{"a":0,"<stop-a>":1,"<stop-b>":2,"<stop-c>":3,"<think>":4,"<stop-d>":5,"<stop-e>":6,"<stop-f>":7,"<stop-g>":8,"<stop-h>":9,"<stop-i>":10,"<stop-j>":11},"merges":[]}}`,
				"config.json":            test.config,
				"generation_config.json": test.generation,
			} {
				if content == "" {
					continue
				}
				if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0600); err != nil {
					t.Fatal(err)
				}
			}
			var config modelConfig
			if err := json.Unmarshal([]byte(test.config), &config); err != nil {
				t.Fatal(err)
			}
			config.ModelType, config.Vocabulary, config.HeadDim = "llama", 12, 2
			config.HiddenSize, config.IntermediateSize = 2, 2
			config.AttentionHeads, config.KVHeads, config.HiddenLayers = 1, 1, 1
			config.MaxPositions, config.RMSEpsilon, config.RopeTheta, config.TiedEmbeddings = 16, 1e-5, 10000, true
			encoded, err := json.Marshal(config)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(directory, "config.json"), encoded, 0600); err != nil {
				t.Fatal(err)
			}
			weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{Vocab: int(config.Vocabulary), Hidden: int(config.HiddenSize), Heads: int(config.AttentionHeads), HeadDim: int(config.HeadDim), KVHeads: int(config.KVHeads), Intermediate: int(config.IntermediateSize), Layers: int(config.HiddenLayers)})
			if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, shapes, nil); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(directory, "converted.gguf")
			_, err = Convert(Options{Directory: directory, OutputPath: path})
			if test.invalid {
				if err == nil {
					t.Fatal("accepted invalid ID after first EOS entry")
				}
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					t.Fatal("invalid stop metadata published a GGUF")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			file, err := gguf.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			vocabulary, err := tokenizer.Load(file)
			if err != nil {
				t.Fatal(err)
			}
			if actual := vocabulary.EOGTokens(); !slices.Equal(actual, test.want) {
				t.Fatalf("converted declared stop IDs = %v; want %v", actual, test.want)
			}
			if vocabulary.EOS != test.primary {
				t.Fatalf("primary EOS=%d, want %d", vocabulary.EOS, test.primary)
			}
			if vocabulary.IsEOG(4) {
				t.Fatal("undeclared <think> token became a stop")
			}
		})
	}
}
