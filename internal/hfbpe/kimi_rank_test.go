package hfbpe

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Colibri c/tools/k3_tokenizer.py at f028d26b422144ed4a69ad9aeaee2553ce0f9572
// derives tiktoken ranks as vocabulary IDs and deliberately emits no merge
// list. These tiny ranks expose a different answer from leftmost merging.
func TestKimiRankBPEDeclaration(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "tokenizer", "testdata", "kimi-declared-split.json"))
	if err != nil {
		t.Fatal(err)
	}
	var source struct {
		SourceSHA256 string `json:"source_sha256"`
		Pattern      string `json:"pattern"`
	}
	if err := json.Unmarshal(raw, &source); err != nil {
		t.Fatal(err)
	}
	if source.SourceSHA256 != "99b5ff4e6cc9f14bcdc9c4e1d854c35d4afc087af8e825f3264b1ebbc10132eb" {
		t.Fatal("Kimi Split source changed")
	}
	model := map[string]any{
		"type": "BPE", "ignore_merges": true, "merges": []string{},
		"vocab": map[string]int{
			"a": 0, "b": 1, "c": 2, "d": 3,
			"bc": 4, "cd": 5, "bcd": 7, "ab": 9, "abc": 11,
			"x": 12, "y": 13, "z": 14, "xy": 15, "yz": 15,
		},
	}
	load := func(pre any) *Tokenizer {
		t.Helper()
		dir := t.TempDir()
		data, err := json.Marshal(map[string]any{"model": model, "pre_tokenizer": pre})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0600); err != nil {
			t.Fatal(err)
		}
		encoder, err := Load(dir)
		if err != nil {
			t.Fatal(err)
		}
		return encoder
	}
	pre := map[string]any{"type": "Sequence", "pretokenizers": []any{
		map[string]any{"type": "Split", "pattern": map[string]string{"Regex": source.Pattern}, "behavior": "Isolated", "invert": false},
		map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
	}}
	encoder := load(pre)
	for _, sample := range []struct {
		text string
		ids  []int
	}{
		{"abc", []int{11}},    // Whole-piece vocabulary shortcut.
		{"abcd", []int{0, 7}}, // bc has the lowest adjacent rank, then bcd.
		{"ad", []int{0, 3}},   // No eligible pair leaves byte symbols.
	} {
		got, err := encoder.Encode(sample.text)
		if err != nil || !slices.Equal(got, sample.ids) {
			t.Errorf("Encode(%q) = %v, %v; want %v", sample.text, got, err, sample.ids)
		}
		if got, err := encoder.DecodeStrict(sample.ids); err != nil || got != sample.text {
			t.Errorf("DecodeStrict(%v) = %q, %v; want %q", sample.ids, got, err, sample.text)
		}
	}
	if got := encoder.bpe("xyz"); !slices.Equal(got, []string{"xy", "z"}) {
		t.Fatalf("equal adjacent ranks chose %v; want leftmost pair", got)
	}
	unknown := load(map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false})
	if _, err := unknown.Encode("abcd"); err == nil || !strings.Contains(err.Error(), "empty-merge rank BPE") {
		t.Fatalf("unrecognized rank-BPE declaration did not refuse: %v", err)
	}
	pre["pretokenizers"] = append(pre["pretokenizers"].([]any)[:1],
		map[string]any{"type": "Digits", "individual_digits": true},
		map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false})
	withExtraStage := load(pre)
	if _, err := withExtraStage.Encode("abcd"); err == nil || !strings.Contains(err.Error(), "empty-merge rank BPE") {
		t.Fatalf("modified Kimi declaration did not refuse: %v", err)
	}
}
