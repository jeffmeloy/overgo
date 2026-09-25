package hfconvert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
)

func TestKimiDeclaredRankConversion(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "hfbpe", "testdata", "kimi_k3_rank_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Pattern     string          `json:"pattern"`
		Vocab       map[string]int  `json:"vocab"`
		AddedTokens json.RawMessage `json:"added_tokens"`
		Cases       []struct {
			Input string `json:"input"`
			IDs   []int  `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	makeSource := func() map[string]any {
		return map[string]any{
			"normalizer": nil,
			"pre_tokenizer": map[string]any{"type": "Sequence", "pretokenizers": []any{
				map[string]any{"type": "Split", "pattern": map[string]string{"Regex": reference.Pattern}, "behavior": "Isolated", "invert": false},
				map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
			}},
			"decoder":      map[string]any{"type": "ByteLevel"},
			"model":        map[string]any{"type": "BPE", "ignore_merges": true, "merges": []string{}, "vocab": reference.Vocab},
			"added_tokens": reference.AddedTokens,
		}
	}
	dir := t.TempDir()
	convert := func(source map[string]any) ([]gguf.Metadata, error) {
		t.Helper()
		data, err := json.Marshal(source)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0600); err != nil {
			t.Fatal(err)
		}
		return tokenizerMetadata(dir, 163588, nil)
	}
	metadata, err := convert(makeSource())
	if err != nil {
		t.Fatal(err)
	}
	vocab, err := tokenizer.Load(&gguf.File{Metadata: metadata})
	if err != nil {
		t.Fatal(err)
	}
	if vocab.Pre != "kimi" || !vocab.IgnoreMerges {
		t.Fatal("converter did not bind Kimi rank BPE to native GGUF metadata")
	}
	for _, index := range []int{0, 2, 8, 13, 14} {
		sample := reference.Cases[index]
		got, err := vocab.Encode(sample.Input, tokenizer.EncodeOptions{})
		want := make([]tokenizer.TokenID, len(sample.IDs))
		for i, id := range sample.IDs {
			want[i] = tokenizer.TokenID(id)
		}
		if err != nil || !slices.Equal(got, want) {
			t.Errorf("converted Kimi Encode(%q) = %v, %v; want %v", sample.Input, got, err, want)
		}
	}
	for _, variant := range []struct {
		name   string
		change func(map[string]any)
	}{
		{"model_type", func(source map[string]any) { source["model"].(map[string]any)["type"] = "Unigram" }},
		{"ignore_merges_false", func(source map[string]any) { source["model"].(map[string]any)["ignore_merges"] = false }},
		{"merges_absent", func(source map[string]any) { delete(source["model"].(map[string]any), "merges") }},
		{"merges_present", func(source map[string]any) { source["model"].(map[string]any)["merges"] = []string{"a b"} }},
		{"split_behavior", func(source map[string]any) {
			source["pre_tokenizer"].(map[string]any)["pretokenizers"].([]any)[0].(map[string]any)["behavior"] = "MergedWithPrevious"
		}},
		{"byte_prefix", func(source map[string]any) {
			source["pre_tokenizer"].(map[string]any)["pretokenizers"].([]any)[1].(map[string]any)["add_prefix_space"] = true
		}},
		{"normalizer", func(source map[string]any) { source["normalizer"] = map[string]string{"type": "NFC"} }},
		{"decoder", func(source map[string]any) { source["decoder"] = map[string]string{"type": "Metaspace"} }},
	} {
		t.Run(variant.name, func(t *testing.T) {
			source := makeSource()
			variant.change(source)
			if _, err := convert(source); err == nil || !strings.Contains(err.Error(), "Kimi") {
				t.Fatalf("unsupported Kimi declaration was accepted: %v", err)
			}
		})
	}
}
