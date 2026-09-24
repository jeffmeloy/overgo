package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gguf"
)

func TestKimiNativeRankBPE(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "hfbpe", "testdata", "kimi_k3_rank_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Vocab map[string]int `json:"vocab"`
		Cases []struct {
			Input string `json:"input"`
			IDs   []int  `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	maxID := 0
	for _, id := range reference.Vocab {
		maxID = max(maxID, id)
	}
	tokens := make([]string, maxID+1)
	types := make([]int32, len(tokens))
	for id := range tokens {
		tokens[id] = fmt.Sprintf("[PAD%d]", id)
		types[id] = int32(TokenUnused)
	}
	for token, id := range reference.Vocab {
		tokens[id] = token
		types[id] = int32(TokenNormal)
	}
	metadata := []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "gpt2"),
		scalar("tokenizer.ggml.pre", gguf.ValueTypeString, "kimi"),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		array("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
		array("tokenizer.ggml.merges", gguf.ValueTypeString, []string{}),
	}
	encoder, err := Load(&gguf.File{Metadata: metadata})
	if err != nil {
		t.Fatal(err)
	}
	if !encoder.IgnoreMerges || !encoder.rankBPE {
		t.Fatal("Kimi GGUF pre-tokenizer did not select rank BPE")
	}
	for index, sample := range reference.Cases {
		got, err := encoder.Encode(sample.Input, EncodeOptions{})
		if err != nil {
			t.Errorf("case %d: %v", index, err)
			continue
		}
		want := make([]TokenID, len(sample.IDs))
		for i, id := range sample.IDs {
			want[i] = TokenID(id)
		}
		if !slices.Equal(got, want) {
			t.Errorf("case %d Encode(%q) = %v, want %v", index, sample.Input, got, want)
		}
		decoded, err := encoder.Decode(want, true)
		if err != nil || decoded != sample.Input {
			t.Errorf("case %d Decode(%v) = %q, %v", index, want, decoded, err)
		}
	}
	bad := slices.Clone(metadata)
	bad[len(bad)-1] = array("tokenizer.ggml.merges", gguf.ValueTypeString, []string{"a b"})
	if _, err := Load(&gguf.File{Metadata: bad}); err == nil || !strings.Contains(err.Error(), "empty GGUF merges") {
		t.Fatalf("nonempty Kimi merges were accepted: %v", err)
	}
}
