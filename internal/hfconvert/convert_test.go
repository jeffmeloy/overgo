package hfconvert

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

func TestDenseConversionPreservesChatTemplate(t *testing.T) {
	directory := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0},"merges":[]}}`), 0o644))
	const template = "{% if enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
	path := filepath.Join(directory, "chat_template.jinja")
	testcheck.NoError(t, os.WriteFile(path, []byte(template), 0o644))
	metadata, err := modelMetadata(directory, "fixture", archProfiles["llama"], modelConfig{Vocabulary: 1, HeadDim: 1})
	testcheck.NoError(t, err)
	count := 0
	for _, item := range metadata {
		if item.Key == "tokenizer.chat_template" {
			count++
			if item.Value.Data != template {
				t.Fatal("template changed")
			}
		}
	}
	if count != 1 {
		t.Fatalf("chat template count=%d", count)
	}
	testcheck.NoError(t, os.Remove(path))
	testcheck.NoError(t, os.Mkdir(path, 0o755))
	if _, err := modelMetadata(directory, "fixture", archProfiles["llama"], modelConfig{Vocabulary: 1, HeadDim: 1}); err == nil {
		t.Fatal("unreadable template was silently dropped")
	}
}

func TestDNATokenizerMetadataPreservesDeclaredExtension(t *testing.T) {
	directory := t.TempDir()
	config := `{
		"k": 2,
		"dna_start_id": 10,
		"dna_vocab_size": 20,
		"dna_special_tokens": ["<dna>", "</dna>", "<oov>"],
		"auto_dna_tags": true
	}`
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "dna_config.json"), []byte(config), 0o644))
	metadata, err := dnaTokenizerMetadata(directory, 32)
	testcheck.NoError(t, err)
	values := make(map[string]any, len(metadata))
	for _, item := range metadata {
		values[item.Key] = item.Value.Data
	}
	if values[tokenizer.MetadataDNAK] != uint32(2) ||
		values[tokenizer.MetadataDNAStartID] != uint32(10) ||
		values[tokenizer.MetadataDNAVocabulary] != uint32(20) ||
		values[tokenizer.MetadataDNAAutoTags] != true {
		t.Fatalf("DNA metadata scalars = %v", values)
	}
	specials, ok := values[tokenizer.MetadataDNASpecialTokens].([]string)
	if !ok || len(specials) != 3 || specials[2] != "<oov>" {
		t.Fatalf("DNA special tokens = %#v", values[tokenizer.MetadataDNASpecialTokens])
	}
}

func TestDNATokenizerMetadataRejectsTruncatedRange(t *testing.T) {
	directory := t.TempDir()
	config := `{"k":2,"dna_start_id":10,"dna_vocab_size":18,"dna_special_tokens":["<dna>","</dna>","<oov>"]}`
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "dna_config.json"), []byte(config), 0o644))
	if _, err := dnaTokenizerMetadata(directory, 32); err == nil {
		t.Fatal("truncated DNA range accepted")
	}
}
