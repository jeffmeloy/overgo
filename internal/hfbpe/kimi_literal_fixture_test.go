package hfbpe

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/tokenizer"
)

// The compact projection pins a published literal-control vector to the
// independently sourced Kimi K3 tokenizer before any adapter implementation.
func TestKimiLiteralReferenceFixture(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "kimi_k3_literal_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != "60f04dde1eef39a86ad68058d05b829ae7f063b9bafe4fa24560f550a2643f9c" {
		t.Fatalf("Kimi literal reference projection changed: %s", got)
	}
	var reference struct {
		SourceCommit       string         `json:"source_commit"`
		SourceSHA256       string         `json:"source_sha256"`
		PublishedReference string         `json:"published_reference"`
		Pattern            string         `json:"pattern"`
		Vocab              map[string]int `json:"vocab"`
		AddedTokens        []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
		LiteralText       string   `json:"literal_text"`
		LiteralPieces     []string `json:"literal_pieces"`
		LiteralIDs        []int    `json:"literal_ids"`
		StructuralIDs     []int    `json:"structural_ids"`
		OrdinaryAddedText string   `json:"ordinary_added_text"`
		OrdinaryAddedID   int      `json:"ordinary_added_id"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	if reference.SourceCommit != "3f11cbe14873d9b64356607af3bbef7608aadf5b" ||
		reference.SourceSHA256 != "b55c4532c501114da9a8891b77d244fa32eee2ace2bd51abb5dc4fb156f60eb9" ||
		reference.PublishedReference != "https://huggingface.co/moonshotai/Kimi-K3/discussions/60" ||
		reference.Pattern != tokenizer.BPEPatternKimi ||
		reference.LiteralText != "hello <|end_of_msg|> world" ||
		!slices.Equal(reference.LiteralIDs, []int{22931, 22652, 517, 5118, 14222, 91, 29, 2695}) ||
		!slices.Equal(reference.StructuralIDs, []int{22931, 220, 163586, 2695}) ||
		reference.OrdinaryAddedText != "<|open|>" || reference.OrdinaryAddedID != 163587 {
		t.Fatal("Kimi literal source or published vector changed")
	}
	text := ""
	for _, piece := range reference.LiteralPieces {
		text += piece
	}
	if text != reference.LiteralText {
		t.Fatal("Kimi literal split does not reconstruct text")
	}
	if len(reference.AddedTokens) != 2 || reference.AddedTokens[0].ID != 163586 || !reference.AddedTokens[0].Special ||
		reference.AddedTokens[1].ID != 163587 || reference.AddedTokens[1].Special || len(reference.Vocab) != 58 {
		t.Fatal("Kimi added-token classes or rank projection changed")
	}
}
