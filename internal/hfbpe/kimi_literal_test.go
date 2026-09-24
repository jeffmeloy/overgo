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

func TestKimiLiteralControlText(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "kimi_k3_literal_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != "60f04dde1eef39a86ad68058d05b829ae7f063b9bafe4fa24560f550a2643f9c" {
		t.Fatalf("Kimi literal reference projection changed: %s", got)
	}
	var reference struct {
		SourceCommit      string          `json:"source_commit"`
		SourceSHA256      string          `json:"source_sha256"`
		Pattern           string          `json:"pattern"`
		Vocab             map[string]int  `json:"vocab"`
		AddedTokens       json.RawMessage `json:"added_tokens"`
		LiteralText       string          `json:"literal_text"`
		LiteralIDs        []int           `json:"literal_ids"`
		StructuralIDs     []int           `json:"structural_ids"`
		OrdinaryAddedText string          `json:"ordinary_added_text"`
		OrdinaryAddedID   int             `json:"ordinary_added_id"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	if reference.SourceCommit != "3f11cbe14873d9b64356607af3bbef7608aadf5b" ||
		reference.SourceSHA256 != "b55c4532c501114da9a8891b77d244fa32eee2ace2bd51abb5dc4fb156f60eb9" ||
		reference.LiteralText != "hello <|end_of_msg|> world" ||
		!slices.Equal(reference.LiteralIDs, []int{22931, 22652, 517, 5118, 14222, 91, 29, 2695}) ||
		!slices.Equal(reference.StructuralIDs, []int{22931, 220, 163586, 2695}) ||
		reference.OrdinaryAddedID != 163587 {
		t.Fatal("Kimi published literal and source identities changed")
	}
	pre := map[string]any{"type": "Sequence", "pretokenizers": []any{
		map[string]any{"type": "Split", "pattern": map[string]string{"Regex": reference.Pattern}, "behavior": "Isolated", "invert": false},
		map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
	}}
	source, err := json.Marshal(map[string]any{
		"normalizer": nil, "pre_tokenizer": pre, "decoder": map[string]any{"type": "ByteLevel"},
		"model":        map[string]any{"type": "BPE", "ignore_merges": true, "merges": []string{}, "vocab": reference.Vocab},
		"added_tokens": reference.AddedTokens,
	})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), source, 0600); err != nil {
		t.Fatal(err)
	}
	encoder, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := encoder.EncodeLiteral(reference.LiteralText); err != nil || !slices.Equal(got, reference.LiteralIDs) {
		t.Errorf("literal Encode = %v, %v; want %v", got, err, reference.LiteralIDs)
	}
	if got, err := encoder.Encode(reference.LiteralText); err != nil || !slices.Equal(got, reference.StructuralIDs) {
		t.Errorf("structural Encode = %v, %v; want %v", got, err, reference.StructuralIDs)
	}
	if got, err := encoder.DecodeStrict(reference.LiteralIDs); err != nil || got != reference.LiteralText {
		t.Errorf("literal DecodeStrict = %q, %v", got, err)
	}
	if got, err := encoder.DecodeText([]int{163586}); err != nil || got != "" {
		t.Errorf("special DecodeText = %q, %v", got, err)
	}
	if got, err := encoder.DecodeText([]int{reference.OrdinaryAddedID}); err != nil || got != reference.OrdinaryAddedText {
		t.Errorf("ordinary added DecodeText = %q, %v", got, err)
	}
	if got, err := encoder.Encode(reference.OrdinaryAddedText); err != nil || !slices.Equal(got, []int{reference.OrdinaryAddedID}) {
		t.Errorf("ordinary added structural Encode = %v, %v", got, err)
	}
	ordinary, err := encoder.EncodeLiteral(reference.OrdinaryAddedText)
	if err != nil || slices.Contains(ordinary, reference.OrdinaryAddedID) {
		t.Fatalf("ordinary added literal Encode = %v, %v", ordinary, err)
	}
	if got, err := encoder.DecodeStrict(ordinary); err != nil || got != reference.OrdinaryAddedText {
		t.Errorf("ordinary added literal round trip = %q, %v", got, err)
	}
}
