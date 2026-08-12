package latentimage

import (
	"fmt"

	"overgo/internal/hfbpe"
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

// KreaTextTemplate holds the Krea conditioner template facts. Every field is a
// CITED reference fact (no inline magic), mirroring adaptive's repodb facts
// text.prompt_template.prefix / .suffix, text.max_prompt_tokens, and the
// tokenizer pad token -- and the routedlm.PromptTemplate sibling contract. overgo
// has no model-fact store, so these live here as a cited contract function.
type KreaTextTemplate struct {
	// Prefix is prepended to the user prompt before tokenization. Verbatim from
	// krea-2/encoder.py:35 Qwen3VLConditioner.prompt_template_encode_prefix
	// (adaptive repodb fact text.prompt_template.prefix).
	Prefix string
	// Suffix is tokenized independently and appended after the pad region.
	// Verbatim from krea-2/encoder.py:36 prompt_template_encode_suffix
	// (adaptive repodb fact text.prompt_template.suffix).
	Suffix string
	// MaxPromptTokens is the prompt/pad row budget between prefix and suffix.
	// krea-2/encoder.py:15 TextEncoderConfig.max_length = 512
	// (adaptive repodb fact text.max_prompt_tokens, sourced from that field).
	MaxPromptTokens int
	// PadToken is the literal filling the unattended pad rows. tokenizer/
	// tokenizer_config.json "pad_token" = "<|endoftext|>"; resolved to an id via
	// the tokenizer at render time (never a hardcoded id).
	PadToken string
	// PrefixTokens / SuffixTokens are the reference conditioner's asserted token
	// counts for the prefix / suffix (krea-2/encoder.py:37-38
	// prompt_template_encode_start_idx = 34, prompt_template_encode_suffix_start_idx
	// = 5). They are NOT used to build the layout (which derives the counts from the
	// actual tokenization, exactly like adaptive); they are the independent anchor
	// the renderer cross-checks the tokenizer against, so a tokenizer/template drift
	// is caught at the template boundary.
	PrefixTokens int
	SuffixTokens int
}

// KreaChatPromptTemplate returns the Krea-2-Turbo conditioner template facts.
func KreaChatPromptTemplate() KreaTextTemplate {
	return KreaTextTemplate{
		Prefix: "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, " +
			"quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
			"<|im_start|>user\n",
		Suffix:          "<|im_end|>\n<|im_start|>assistant\n",
		MaxPromptTokens: 512,
		PadToken:        "<|endoftext|>",
		PrefixTokens:    34,
		SuffixTokens:    5,
	}
}

// KreaTextInput is the rendered conditioner input: the full padded id sequence,
// its attention mask, and PromptRows -- the count the encoder keeps after dropping
// the prefix (len(IDs) - len(prefix tokens) == MaxPromptTokens). IDs/Mask both have
// length MaxPromptTokens + len(prefix tokens).
type KreaTextInput struct {
	IDs        []int
	Mask       []bool
	PromptRows int
}

// RenderKreaTextInput builds the exact templated ids + attention mask the Krea
// Qwen3-VL conditioner consumes. tok is the model's Qwen2 byte-level BPE tokenizer
// (hfbpe.Load over tokenizer.json). Ports adaptive prepareSelectedLayerTextInput.
func RenderKreaTextInput(tok *hfbpe.Tokenizer, prompt string, tmpl KreaTextTemplate) (KreaTextInput, error) {
	var out KreaTextInput
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
	return assembleKreaRows(prefixIDs, promptIDs, suffixIDs, tmpl.MaxPromptTokens, padID), nil
}

// assembleKreaRows builds the fixed-row [prefix][prompt][pad...][suffix] id
// sequence + attention mask from the already-tokenized parts. Pure port of the
// adaptive prepareSelectedLayerTextInput body (no tokenizer), so the layout math
// is exercised in CI without the checkpoint. The prompt is truncated to
// maxPromptTokens; padded = maxPromptTokens + len(prefix) - len(suffix); the pad
// region between prompt and suffix is unattended (mask false).
func assembleKreaRows(prefixIDs, promptIDs, suffixIDs []int, maxPromptTokens, padID int) KreaTextInput {
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
	return KreaTextInput{IDs: ids, Mask: mask, PromptRows: len(ids) - len(prefixIDs)}
}
