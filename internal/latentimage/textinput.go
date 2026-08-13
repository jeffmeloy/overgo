package latentimage

import (
	"fmt"

	"overgo/internal/hfbpe"
	"overgo/internal/modelrecipe"
)

// Krea text-input renderer: prompt string -> Qwen2 BPE ids -> chat prompt-template
// prefix/suffix wrap -> pad to a fixed row count -> ids + attention mask. This is
// the DEVICE-text-conditioning input boundary (dtc brick 1/3): the exact templated
// input the Krea Qwen3-VL conditioner consumes, which the raw tokenized-ids path in
// textencoder.go does NOT reproduce (see the ORACLE SCOPE note there).
//
// Ported VERBATIM from adaptive_new go/extmodel:
//   - models.go prepareSelectedLayerTextInput (the fixed-row prefix/prompt/pad/
//     suffix layout + attention mask), and
//   - models.go loadSelectedLayerTextEncoderConfig (prefix/suffix are tokenized
//     independently; pad row id resolved from the tokenizer_config pad_token).
//
// The Krea conditioner reference (models/Krea-2-Turbo/krea-2/encoder.py
// Qwen3VLConditioner.forward) tokenizes prefix+prompt with padding="max_length" to
//   max_length + prompt_template_encode_start_idx - prompt_template_encode_suffix_start_idx
// then concatenates the separately-tokenized suffix. With RIGHT padding this is the
// exact [prefix][prompt][pad...][suffix] layout adaptive reproduces; the encoder
// then drops the first prefix rows (hiddens[:, prefix_idx:]).

// TextInput is the rendered conditioner input: the full padded id sequence,
// its attention mask, and PromptRows -- the count the encoder keeps after dropping
// the prefix (len(IDs) - len(prefix tokens) == MaxPromptTokens). IDs/Mask both have
// length MaxPromptTokens + len(prefix tokens).
type TextInput struct {
	IDs        []int
	Mask       []bool
	PromptRows int
}

// RenderTextInput builds profile-owned templated ids + attention mask.
func RenderTextInput(tok *hfbpe.Tokenizer, prompt string, tmpl modelrecipe.ImageConditioning) (TextInput, error) {
	var out TextInput
	if tmpl.Prefix == "" || tmpl.Suffix == "" {
		return out, fmt.Errorf("krea text template prefix/suffix is empty")
	}
	if tmpl.MaxPromptTokens <= 0 {
		return out, fmt.Errorf("krea text template max_prompt_tokens must be positive, got %d", tmpl.MaxPromptTokens)
	}
	padID, ok := tok.SpecialID(tmpl.PadToken)
	if !ok {
		return out, fmt.Errorf("krea pad token %q is not in tokenizer", tmpl.PadToken)
	}
	prefixIDs, err := tok.Encode(tmpl.Prefix)
	if err != nil {
		return out, fmt.Errorf("encode krea template prefix: %w", err)
	}
	suffixIDs, err := tok.Encode(tmpl.Suffix)
	if err != nil {
		return out, fmt.Errorf("encode krea template suffix: %w", err)
	}
	// Cross-check the tokenization against the reference conditioner's asserted
	// token counts (encoder.py prompt_template_encode_start_idx / suffix_start_idx).
	// A mismatch means the tokenizer or the template strings drifted from the golden.
	if tmpl.PrefixTokens != 0 && len(prefixIDs) != tmpl.PrefixTokens {
		return out, fmt.Errorf("krea prefix tokenized to %d ids, reference asserts %d", len(prefixIDs), tmpl.PrefixTokens)
	}
	if tmpl.SuffixTokens != 0 && len(suffixIDs) != tmpl.SuffixTokens {
		return out, fmt.Errorf("krea suffix tokenized to %d ids, reference asserts %d", len(suffixIDs), tmpl.SuffixTokens)
	}
	if len(prefixIDs) == 0 || len(suffixIDs) == 0 {
		return out, fmt.Errorf("krea template prefix/suffix tokenized empty")
	}
	promptIDs, err := tok.Encode(prompt)
	if err != nil {
		return out, fmt.Errorf("encode krea prompt: %w", err)
	}
	return assembleTextRows(prefixIDs, promptIDs, suffixIDs, tmpl.MaxPromptTokens, padID), nil
}

// assembleTextRows builds the fixed-row [prefix][prompt][pad...][suffix] id
// sequence + attention mask from the already-tokenized parts. Pure port of the
// adaptive prepareSelectedLayerTextInput body (no tokenizer), so the layout math
// is exercised in CI without the checkpoint. The prompt is truncated to
// maxPromptTokens; padded = maxPromptTokens + len(prefix) - len(suffix); the pad
// region between prompt and suffix is unattended (mask false).
func assembleTextRows(prefixIDs, promptIDs, suffixIDs []int, maxPromptTokens, padID int) TextInput {
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
	return TextInput{IDs: ids, Mask: mask, PromptRows: len(ids) - len(prefixIDs)}
}
