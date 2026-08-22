package latentimage

import (
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/hfbpe"
	"overgo/internal/representation"
)

// Profile-driven text conditioning: template, fixed rows, pad mask.

// textTemplate: recipe-bound conditioner facts.
type textTemplate struct {
	// Prefix is prepended to the user prompt before tokenization. Verbatim from
	// krea-2/encoder.py:35 Qwen3VLConditioner.prompt_template_encode_prefix
	// (adaptive repodb fact text.prompt_template.prefix).
	Prefix string `json:"prefix"`
	// Suffix is tokenized independently and appended after the pad region.
	// Verbatim from krea-2/encoder.py:36 prompt_template_encode_suffix
	// (adaptive repodb fact text.prompt_template.suffix).
	Suffix string `json:"suffix"`
	// MaxPromptTokens is the prompt/pad row budget between prefix and suffix.
	// krea-2/encoder.py:15 TextEncoderConfig.max_length = 512
	// (adaptive repodb fact text.max_prompt_tokens, sourced from that field).
	MaxTokens int `json:"max_tokens"`
	// PadToken is the literal filling the unattended pad rows. tokenizer/
	// tokenizer_config.json "pad_token" = "<|endoftext|>"; resolved to an id via
	// the tokenizer at render time (never a hardcoded id).
	PadToken string `json:"pad_token"`
	// PrefixTokens / SuffixTokens are the reference conditioner's asserted token
	// counts for the prefix / suffix (krea-2/encoder.py:37-38
	// prompt_template_encode_start_idx = 34, prompt_template_encode_suffix_start_idx
	// = 5). They are NOT used to build the layout (which derives the counts from the
	// actual tokenization, exactly like adaptive); they are the independent anchor
	// the renderer cross-checks the tokenizer against, so a tokenizer/template drift
	// is caught at the template boundary.
	PrefixTokens int `json:"prefix_tokens"`
	SuffixTokens int `json:"suffix_tokens"`
}

// textInput: rendered conditioner rows and mask.
type textInput struct {
	IDs        []int
	Mask       []bool
	PromptRows int
}

// renderTextInput builds templated ids plus the attention mask.
func renderTextInput(tok *hfbpe.Tokenizer, prompt string, tmpl textTemplate) (textInput, error) {
	var out textInput
	if !checked.NonzeroAll(tmpl.Prefix, tmpl.Suffix) {
		return out, fmt.Errorf("image text template prefix/suffix is empty")
	}
	if !checked.PositiveInts(tmpl.MaxTokens) {
		return out, fmt.Errorf("image text template max_tokens must be positive, got %d", tmpl.MaxTokens)
	}
	padID, ok := tok.SpecialID(tmpl.PadToken)
	if !ok {
		return out, fmt.Errorf("image pad token %q is not in tokenizer", tmpl.PadToken)
	}
	prefixIDs, err := tok.Encode(tmpl.Prefix)
	if err != nil {
		return out, fmt.Errorf("encode image template prefix: %w", err)
	}
	suffixIDs, err := tok.Encode(tmpl.Suffix)
	if err != nil {
		return out, fmt.Errorf("encode image template suffix: %w", err)
	}
	// Cross-check the tokenization against the reference conditioner's asserted
	// token counts (encoder.py prompt_template_encode_start_idx / suffix_start_idx).
	// A mismatch means the tokenizer or the template strings drifted from the golden.
	if checked.Nonzero(tmpl.PrefixTokens) && !checked.Equal(len(prefixIDs), tmpl.PrefixTokens) {
		return out, fmt.Errorf("image prefix tokenized to %d ids, profile asserts %d", len(prefixIDs), tmpl.PrefixTokens)
	}
	if checked.Nonzero(tmpl.SuffixTokens) && !checked.Equal(len(suffixIDs), tmpl.SuffixTokens) {
		return out, fmt.Errorf("image suffix tokenized to %d ids, profile asserts %d", len(suffixIDs), tmpl.SuffixTokens)
	}
	if !checked.NonemptyAll(prefixIDs, suffixIDs) {
		return out, fmt.Errorf("image template prefix/suffix tokenized empty")
	}
	promptIDs, err := tok.Encode(prompt)
	if err != nil {
		return out, fmt.Errorf("encode image prompt: %w", err)
	}
	sequence, err := representation.AssemblePaddedSequence(prefixIDs, promptIDs, suffixIDs, tmpl.MaxTokens, padID)
	if err != nil {
		return out, fmt.Errorf("image text sequence: %w", err)
	}
	return textInput{IDs: sequence.IDs, Mask: sequence.Mask, PromptRows: sequence.PromptRows}, nil
}
