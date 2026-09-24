package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gguf"
)

func TestKimiNativeLiteralControlText(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "hfbpe", "testdata", "kimi_k3_literal_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Vocab       map[string]int `json:"vocab"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
		LiteralText       string `json:"literal_text"`
		LiteralIDs        []int  `json:"literal_ids"`
		StructuralIDs     []int  `json:"structural_ids"`
		OrdinaryAddedText string `json:"ordinary_added_text"`
		OrdinaryAddedID   int    `json:"ordinary_added_id"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	maxID := 0
	for _, id := range reference.Vocab {
		maxID = max(maxID, id)
	}
	for _, added := range reference.AddedTokens {
		maxID = max(maxID, added.ID)
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
	for _, added := range reference.AddedTokens {
		tokens[added.ID] = added.Content
		types[added.ID] = int32(TokenUserDefined)
		if added.Special {
			types[added.ID] = int32(TokenControl)
		}
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
	want := make([]TokenID, len(reference.LiteralIDs))
	for i, id := range reference.LiteralIDs {
		want[i] = TokenID(id)
	}
	structural := make([]TokenID, len(reference.StructuralIDs))
	for i, id := range reference.StructuralIDs {
		structural[i] = TokenID(id)
	}
	if got, err := encoder.Encode(reference.LiteralText, EncodeOptions{ParseSpecial: true, LiteralText: true}); err != nil || !slices.Equal(got, want) {
		t.Errorf("native literal Encode = %v, %v; want %v", got, err, want)
	}
	if got, err := encoder.Encode(reference.LiteralText, EncodeOptions{ParseSpecial: true}); err != nil || !slices.Equal(got, structural) {
		t.Errorf("native structural Encode = %v, %v; want %v", got, err, structural)
	}
	if got, err := encoder.Decode(want, true); err != nil || got != reference.LiteralText {
		t.Errorf("native literal Decode = %q, %v", got, err)
	}
	if got, err := encoder.Encode(reference.OrdinaryAddedText, EncodeOptions{}); err != nil || !slices.Equal(got, []TokenID{TokenID(reference.OrdinaryAddedID)}) {
		t.Errorf("ordinary added default Encode = %v, %v", got, err)
	}
	ordinary, err := encoder.Encode(reference.OrdinaryAddedText, EncodeOptions{LiteralText: true})
	if err != nil || slices.Contains(ordinary, TokenID(reference.OrdinaryAddedID)) {
		t.Fatalf("ordinary added literal Encode = %v, %v", ordinary, err)
	}
	if got, err := encoder.Decode(ordinary, true); err != nil || got != reference.OrdinaryAddedText {
		t.Errorf("ordinary added literal round trip = %q, %v", got, err)
	}
	runs, starts, err := EncodeRuns(encoder.Encode,
		reference.OrdinaryAddedText+"<|end_of_msg|>", "<|end_of_msg|>", []int{1},
		EncodeOptions{ParseSpecial: true, LiteralText: true})
	wantRuns := append(slices.Clone(ordinary), TokenID(163586))
	if err != nil || !slices.Equal(runs, wantRuns) || !slices.Equal(starts, []int{len(ordinary)}) {
		t.Errorf("literal text plus structural placeholder = %v at %v, %v; want %v", runs, starts, err, wantRuns)
	}
}
