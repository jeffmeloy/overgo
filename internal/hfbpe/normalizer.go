package hfbpe

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Retain unsupported encoding declarations as errors on Encode, because
// decode-only speech consumers do not need to compile a normalizer.
func (t *Tokenizer) configureNormalizer(raw json.RawMessage) {
	if text := strings.TrimSpace(string(raw)); text == "" || text == "null" {
		return
	}
	var declaration struct {
		Type    string `json:"type"`
		Pattern struct {
			String *string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &declaration); err != nil {
		t.normalizerErr = fmt.Errorf("invalid normalizer declaration: %w", err)
		return
	}
	switch declaration.Type {
	case "NFC":
		t.normalizeNFC = true
	case "Replace":
		// Preserve the established sentencepiece-marker encoding contract.
		if declaration.Pattern.String != nil && *declaration.Pattern.String == " " && declaration.Content != "" {
			t.spaceMarker = declaration.Content
			return
		}
		t.normalizerErr = fmt.Errorf("unsupported Replace normalizer encoding declaration")
	default:
		t.normalizerErr = fmt.Errorf("unsupported normalizer %q for encoding", declaration.Type)
	}
}
