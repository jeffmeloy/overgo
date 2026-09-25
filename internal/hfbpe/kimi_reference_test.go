package hfbpe

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// The compact projection was generated from the full 19.6 MB tokenizer.json
// at Xenova/Kimi-K3-tokenizer commit 3f11cbe14873d9b64356607af3bbef7608aadf5b.
// It retains every vocabulary entry that can affect these fixed pieces.
func TestKimiDeclaredTokenizerReference(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "kimi_k3_rank_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != "fe1bec80dcf4a56ac59c1fffe529d8ea84e7118fe69a941dc3242ca087059ae6" {
		t.Fatalf("Kimi reference projection changed: %s", got)
	}
	var reference struct {
		Source       string          `json:"source"`
		SourceCommit string          `json:"source_commit"`
		SourceSHA256 string          `json:"source_sha256"`
		SplitSHA256  string          `json:"split_source_sha256"`
		Pattern      string          `json:"pattern"`
		Vocab        map[string]int  `json:"vocab"`
		AddedTokens  json.RawMessage `json:"added_tokens"`
		Cases        []struct {
			Input  string   `json:"input"`
			Pieces []string `json:"pieces"`
			IDs    []int    `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	if reference.Source != "Xenova/Kimi-K3-tokenizer tokenizer.json" ||
		reference.SourceCommit != "3f11cbe14873d9b64356607af3bbef7608aadf5b" ||
		reference.SourceSHA256 != "b55c4532c501114da9a8891b77d244fa32eee2ace2bd51abb5dc4fb156f60eb9" ||
		reference.SplitSHA256 != "99b5ff4e6cc9f14bcdc9c4e1d854c35d4afc087af8e825f3264b1ebbc10132eb" ||
		len(reference.Vocab) != 150 || len(reference.Cases) != 15 {
		t.Fatal("Kimi source identity or reference coverage changed")
	}
	splitRaw, err := os.ReadFile(filepath.Join("..", "tokenizer", "testdata", "kimi-declared-split.json"))
	if err != nil {
		t.Fatal(err)
	}
	var split struct {
		Pattern string `json:"pattern"`
	}
	if err := json.Unmarshal(splitRaw, &split); err != nil || reference.Pattern != split.Pattern {
		t.Fatal("Kimi rank oracle uses a different declared Split")
	}
	pre := map[string]any{"type": "Sequence", "pretokenizers": []any{
		map[string]any{"type": "Split", "pattern": map[string]string{"Regex": reference.Pattern}, "behavior": "Isolated", "invert": false},
		map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
	}}
	model := map[string]any{"type": "BPE", "ignore_merges": true, "merges": []string{}, "vocab": reference.Vocab}
	data, err := json.Marshal(map[string]any{
		"model": model, "pre_tokenizer": pre, "decoder": map[string]any{"type": "ByteLevel"},
		"added_tokens": reference.AddedTokens,
	})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	encoder, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	for index, sample := range reference.Cases {
		if got := encoder.preTokenize(sample.Input); !slices.Equal(got, sample.Pieces) {
			t.Errorf("case %d Split(%q) = %q; want %q", index, sample.Input, got, sample.Pieces)
		}
		got, err := encoder.Encode(sample.Input)
		if err != nil || !slices.Equal(got, sample.IDs) {
			t.Errorf("case %d Encode(%q) = %v, %v; want %v", index, sample.Input, got, err, sample.IDs)
		}
		decoded, err := encoder.DecodeStrict(sample.IDs)
		if err != nil || decoded != sample.Input {
			t.Errorf("case %d DecodeStrict(%v) = %q, %v; want %q", index, sample.IDs, decoded, err, sample.Input)
		}
	}
	for _, special := range []struct {
		text string
		id   int
		omit bool
	}{
		{"[BOS]", 163584, true},
		{"<|end_of_msg|>", 163586, true},
		{"<|open|>", 163587, false},
	} {
		id, ok := encoder.SpecialID(special.text)
		if !ok || id != special.id {
			t.Errorf("SpecialID(%q) = %d, %t; want %d", special.text, id, ok, special.id)
		}
		got, err := encoder.Encode(special.text)
		if err != nil || !slices.Equal(got, []int{special.id}) {
			t.Errorf("Encode(%q) = %v, %v; want %d", special.text, got, err, special.id)
		}
		decoded, err := encoder.DecodeStrict([]int{special.id})
		if err != nil || decoded != special.text {
			t.Errorf("DecodeStrict(%d) = %q, %v", special.id, decoded, err)
		}
		visible, err := encoder.DecodeText([]int{special.id})
		if special.omit && (err != nil || visible != "") ||
			!special.omit && (err != nil || visible != special.text) {
			t.Errorf("DecodeText(%d) = %q, %v; omit=%t", special.id, visible, err, special.omit)
		}
	}
}
