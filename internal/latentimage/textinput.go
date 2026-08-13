package latentimage

import (
	"fmt"

	"overgo/internal/hfbpe"
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
	if tmpl.Prefix == "" || tmpl.Suffix == "" {
		return out, fmt.Errorf("image text template prefix/suffix is empty")
	}
	if tmpl.MaxTokens <= 0 {
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
	if tmpl.PrefixTokens != 0 && len(prefixIDs) != tmpl.PrefixTokens {
		return out, fmt.Errorf("image prefix tokenized to %d ids, profile asserts %d", len(prefixIDs), tmpl.PrefixTokens)
	}
	if tmpl.SuffixTokens != 0 && len(suffixIDs) != tmpl.SuffixTokens {
		return out, fmt.Errorf("image suffix tokenized to %d ids, profile asserts %d", len(suffixIDs), tmpl.SuffixTokens)
	}
	if len(prefixIDs) == 0 || len(suffixIDs) == 0 {
		return out, fmt.Errorf("image template prefix/suffix tokenized empty")
	}
	promptIDs, err := tok.Encode(prompt)
	if err != nil {
		return out, fmt.Errorf("encode image prompt: %w", err)
	}
	return assembleTextRows(prefixIDs, promptIDs, suffixIDs, tmpl.MaxTokens, padID), nil
}

// assembleKreaRows builds the fixed-row [prefix][prompt][pad...][suffix] id
// sequence + attention mask from the already-tokenized parts. Pure port of the
// adaptive prepareSelectedLayerTextInput body (no tokenizer), so the layout math
// is exercised in CI without the checkpoint. The prompt is truncated to
// maxPromptTokens; padded = maxPromptTokens + len(prefix) - len(suffix); the pad
// region between prompt and suffix is unattended (mask false).
func assembleTextRows(prefixIDs, promptIDs, suffixIDs []int, maxPromptTokens, padID int) textInput {
	if len(promptIDs) > maxPromptTokens {
		promptIDs = promptIDs[:maxPromptTokens]
	}
	padded := maxPromptTokens + len(prefixIDs) - len(suffixIDs)
	ids := make([]int, padded+len(suffixIDs))
	mask := make([]bool, len(ids))
	for i := range ids {
		ids[i] = padID
	}
	n := copy(ids, prefixIDs)
	n += copy(ids[n:], promptIDs)
	for i := 0; i < n; i++ {
		mask[i] = true
	}
	copy(ids[padded:], suffixIDs)
	for i := padded; i < len(ids); i++ {
		mask[i] = true
	}
	return textInput{IDs: ids, Mask: mask, PromptRows: len(ids) - len(prefixIDs)}
}
