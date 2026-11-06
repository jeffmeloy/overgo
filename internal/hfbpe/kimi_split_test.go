package hfbpe

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

func TestKimiDeclaredSplitIntegration(t *testing.T) {
	raw := testcheck.Must(os.ReadFile(filepath.Join("..", "tokenizer", "testdata", "kimi-declared-split.json")))(t)
	var source struct {
		SourceSHA256 string `json:"source_sha256"`
		Pattern      string `json:"pattern"`
	}
	testcheck.NoError(t, json.Unmarshal(raw, &source))
	if source.SourceSHA256 != "99b5ff4e6cc9f14bcdc9c4e1d854c35d4afc087af8e825f3264b1ebbc10132eb" {
		t.Fatal("Kimi declaration source changed")
	}
	alphabet := &Tokenizer{}
	alphabet.buildByteAlphabet()
	input := "中文abc"
	vocab := make(map[string]int)
	want := make([]int, 0, len(input))
	for _, value := range []byte(input) {
		symbol := string(alphabet.b2u[value])
		id, found := vocab[symbol]
		if !found {
			id = len(vocab)
			vocab[symbol] = id
		}
		want = append(want, id)
	}
	wen := []byte("文")
	last := string(alphabet.b2u[wen[len(wen)-1]])
	a := string(alphabet.b2u['a'])
	mergedID := len(vocab)
	vocab[last+a] = mergedID
	model := map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{last + " " + a}}
	declaration := func(pattern string) any {
		return map[string]any{"type": "Sequence", "pretokenizers": []any{
			map[string]any{"type": "Split", "pattern": map[string]string{"Regex": pattern}, "behavior": "Isolated", "invert": false},
			map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
		}}
	}
	load := func(pre any) *Tokenizer {
		t.Helper()
		directory := t.TempDir()
		data := testcheck.Must(json.Marshal(map[string]any{"model": model, "pre_tokenizer": pre}))(t)
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "tokenizer.json"), data, 0600))
		tok := testcheck.Must(Load(directory))(t)
		return tok
	}
	declared := load(declaration(source.Pattern))
	got, err := declared.Encode(input)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("declared Kimi split ids=%v want=%v error=%v", got, want, err)
	}
	legacy, err := load(nil).Encode(input)
	if err != nil || !slices.Contains(legacy, mergedID) || slices.Equal(legacy, got) {
		t.Fatalf("boundary-crossing merge was not distinguished: legacy=%v declared=%v error=%v", legacy, got, err)
	}
	unknown := load(declaration(strings.Replace(source.Pattern, `\p{Han}`, `\p{L}`, 1)))
	if _, err := unknown.Encode(input); err == nil || !strings.Contains(err.Error(), "unsupported BPE split expression") {
		t.Fatalf("modified declaration did not refuse: %v", err)
	}
}
