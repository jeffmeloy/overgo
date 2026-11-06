package hfbpe

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestAnEmptyAddedTokenIsRefused holds encoding to a refusal when the added vocabulary holds an empty token, which
// both loaders admit when nothing else is declared: the split around it never shortens its span. Load's legacy form and
// LoadSplit's added_tokens.json each refuse, and literal encoding, which parses no added token, still encodes.
func TestAnEmptyAddedTokenIsRefused(t *testing.T) {
	t.Parallel()
	loaded := loadTokenizerFixture(t, nil, nil, []map[string]any{{"id": 256, "content": ""}})
	if ids, err := loaded.Encode("x"); err == nil {
		t.Fatalf("Load's tokenizer with an empty added token encoded %v, want a refusal", ids)
	}
	if _, err := loaded.encode("x", false); err != nil {
		t.Fatalf("literal encoding refused: %v", err)
	}
	directory := t.TempDir()
	for name, content := range map[string]string{
		"vocab.json": `{"x":0}`, "merges.txt": "#version: 0.2\n", "added_tokens.json": `{"":1}`,
	} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600))
	}
	split := testcheck.Must(LoadSplit(directory))(t)
	if ids, err := split.Encode("x"); err == nil {
		t.Fatalf("LoadSplit's tokenizer with an empty added token encoded %v, want a refusal", ids)
	}
}
