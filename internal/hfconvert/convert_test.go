package hfconvert

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/tokenizer"
)

func TestDNATokenizerMetadataPreservesDeclaredExtension(t *testing.T) {
	directory := t.TempDir()
	config := `{
		"k": 2,
		"dna_start_id": 10,
		"dna_vocab_size": 20,
		"dna_special_tokens": ["<dna>", "</dna>", "<oov>"],
		"auto_dna_tags": true
	}`
	if err := os.WriteFile(filepath.Join(directory, "dna_config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	metadata, err := dnaTokenizerMetadata(directory, 32)
	if err != nil {
		t.Fatal(err)
	}
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
	if err := os.WriteFile(filepath.Join(directory, "dna_config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := dnaTokenizerMetadata(directory, 32); err == nil {
		t.Fatal("truncated DNA range accepted")
	}
}
