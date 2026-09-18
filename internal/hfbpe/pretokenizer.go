package hfbpe

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"overgo/internal/tokenizer"
)

// The compiled stages retain raw UTF-8 until the single final ByteLevel stage;
// Encode then applies the existing byte alphabet once. No stage may follow it.
type preTokenizerStage func(string) []string

func (t *Tokenizer) configurePreTokenizer(raw json.RawMessage) {
	if text := strings.TrimSpace(string(raw)); text == "" || text == "null" {
		return // Preserve the explicit legacy Load/LoadSplit contract.
	}
	var stages []preTokenizerStage
	byteLevel := false
	t.preTokenizerErr = compilePreTokenizer(raw, &stages, &byteLevel)
	if t.preTokenizerErr == nil && !byteLevel {
		t.preTokenizerErr = fmt.Errorf("declared pre-tokenizer requires a final ByteLevel stage")
	}
	if t.spaceMarker != "" {
		t.preTokenizerErr = fmt.Errorf("declared byte-level pre-tokenizer cannot use legacy marker normalization")
	}
	t.preTokenize = func(text string) []string {
		parts := []string{text}
		for _, stage := range stages {
			var next []string
			for _, part := range parts {
				if part != "" {
					next = append(next, stage(part)...)
				}
			}
			parts = next
		}
		return parts
	}
}

func compilePreTokenizer(raw json.RawMessage, stages *[]preTokenizerStage, byteLevel *bool) error {
	if *byteLevel {
		return fmt.Errorf("pre-tokenizer stage follows final ByteLevel")
	}
	var node struct {
		Type             string            `json:"type"`
		PreTokenizers    []json.RawMessage `json:"pretokenizers"`
		AddPrefixSpace   *bool             `json:"add_prefix_space"`
		TrimOffsets      *bool             `json:"trim_offsets"`
		UseRegex         *bool             `json:"use_regex"`
		IndividualDigits *bool             `json:"individual_digits"`
		Pattern          map[string]string `json:"pattern"`
		Behavior         string            `json:"behavior"`
		Invert           *bool             `json:"invert"`
	}
	defaultRegex := true
	node.UseRegex = &defaultRegex
	if err := json.Unmarshal(raw, &node); err != nil {
		return fmt.Errorf("invalid pre-tokenizer: %w", err)
	}
	switch node.Type {
	case "Sequence":
		if len(node.PreTokenizers) == 0 {
			return fmt.Errorf("empty pre-tokenizer Sequence")
		}
		for _, child := range node.PreTokenizers {
			if err := compilePreTokenizer(child, stages, byteLevel); err != nil {
				return err
			}
		}
	case "ByteLevel":
		if node.AddPrefixSpace == nil || node.TrimOffsets == nil || node.UseRegex == nil {
			return fmt.Errorf("byte-level pre-tokenizer requires boolean add_prefix_space, trim_offsets and use_regex (when present)")
		}
		// HF defaults only omitted use_regex during deserialization. trim_offsets
		// changes offsets, which Encode does not expose; it cannot change IDs.
		useRegex := *node.UseRegex
		split, err := tokenizer.CompileBPESplit(tokenizer.BPEPatternGPT2)
		if err != nil {
			return err
		}
		*stages = append(*stages, func(text string) []string {
			if *node.AddPrefixSpace && !strings.HasPrefix(text, " ") {
				text = " " + text
			}
			if useRegex {
				return split(text)
			}
			return []string{text}
		})
		*byteLevel = true
	case "Split":
		if node.Invert == nil || len(node.Pattern) != 1 {
			return fmt.Errorf("split pre-tokenizer requires a single regex pattern and explicit invert")
		}
		// Each supported expression covers all input. Both Isolated and
		// inverted Removed preserve exactly its individual regex matches.
		if !(node.Behavior == "Isolated" && !*node.Invert || node.Behavior == "Removed" && *node.Invert) {
			return fmt.Errorf("unsupported Split delimiter behavior")
		}
		split, err := tokenizer.CompileBPESplit(node.Pattern["Regex"])
		if err != nil {
			return err
		}
		*stages = append(*stages, split)
	case "Digits":
		if node.IndividualDigits == nil {
			return fmt.Errorf("digits pre-tokenizer requires individual_digits")
		}
		*stages = append(*stages, func(text string) []string { return splitDigits(text, *node.IndividualDigits) })
	default:
		return fmt.Errorf("unsupported pre-tokenizer %q", node.Type)
	}
	return nil
}

func splitDigits(text string, individual bool) []string {
	var parts []string
	start := 0
	previousNumber := false
	for index, value := range text {
		number := unicode.IsNumber(value)
		if index > start && (number != previousNumber || individual && number) {
			parts = append(parts, text[start:index])
			start = index
		}
		previousNumber = number
	}
	if start < len(text) {
		parts = append(parts, text[start:])
	}
	return parts
}
