package latentimage

import (
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/hfbpe"
	"overgo/internal/jinja"
	"overgo/internal/representation"
)

// Profile-driven text conditioning: template, fixed rows, pad mask.

// textTemplate: recipe-bound conditioner facts.
type textTemplate struct {
	// ChatTemplate preserves an artifact's unpadded chat formatting and whole-sequence truncation.
	ChatTemplate string `json:"chat_template,omitempty"`
	// EnableThinking binds the template option selected by the conditioning pipeline.
	EnableThinking bool `json:"enable_thinking,omitzero"`
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
	// Weights are each row's attention-value weight (weightedSpans), none for a prompt that weights nothing.
	Weights []float32
}

// renderTextInput builds templated ids plus the attention mask.
func renderTextInput(tok *hfbpe.Tokenizer, prompt string, tmpl textTemplate) (textInput, error) {
	var out textInput
	if tok == nil || tmpl.MaxTokens <= 0 {
		return out, fmt.Errorf("image conditioning requires a tokenizer and positive max_tokens, got %d", tmpl.MaxTokens)
	}
	if tmpl.ChatTemplate != "" {
		render := func(content string) (string, error) {
			env := jinja.New()
			env.SetLimits(jinja.ChatLimits())
			formatted, err := env.Render(tmpl.ChatTemplate, map[string]any{
				"messages":              []any{map[string]any{"role": "user", "content": content}},
				"add_generation_prompt": true, "enable_thinking": tmpl.EnableThinking,
				"tools": jinja.None, "documents": jinja.None,
			})
			if err != nil {
				return "", fmt.Errorf("image conditioning template: %w", err)
			}
			return formatted, nil
		}
		tokens, err := chatTokens(prompt, render, tok.Encode)
		if err != nil {
			return out, err
		}
		ids := tokens.ids[:min(len(tokens.ids), tmpl.MaxTokens)]
		if len(ids) == 0 {
			return out, fmt.Errorf("image conditioning template produced no tokens")
		}
		mask := make([]bool, len(ids))
		for i := range mask {
			mask[i] = true
		}
		var weights []float32
		if tokens.weights != nil {
			weights = tokens.weights[:len(ids)]
		}
		return textInput{IDs: ids, Mask: mask, PromptRows: len(ids), Weights: weights}, nil
	}
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
	// A prompt that weights stretches of itself is tokenized a stretch at a time, so each token's weight is known; one
	// that weights nothing is tokenized whole, as it always was.
	spans, err := weightedSpans(prompt)
	if err != nil {
		return out, err
	}
	var tokens weightedTokens
	if spans == nil {
		tokens.ids, err = tok.Encode(prompt)
	} else {
		tokens, err = spanTokens(spans, tok.Encode)
	}
	promptIDs, promptWeights := tokens.ids, tokens.weights
	if err != nil {
		return out, fmt.Errorf("encode image prompt: %w", err)
	}
	sequence, err := representation.AssemblePaddedSequence(prefixIDs, promptIDs, suffixIDs, tmpl.MaxTokens, padID)
	if err != nil {
		return out, fmt.Errorf("image text sequence: %w", err)
	}
	return textInput{IDs: sequence.IDs, Mask: sequence.Mask, PromptRows: sequence.PromptRows, Weights: rowWeights(len(sequence.IDs), len(prefixIDs), sequence.PromptRows, promptWeights)}, nil
}

// rowWeights are the weights of rows rows of which the prompt's start after prefix and fill at most promptRows, the
// prompt's tokens weighing promptWeights and every other row one; none for a prompt that weights nothing.
func rowWeights(rows, prefix, promptRows int, promptWeights []float32) []float32 {
	if promptWeights == nil {
		return nil
	}
	weights := make([]float32, rows)
	for index := range weights {
		weights[index] = unweighted
	}
	for index, weight := range promptWeights[:min(len(promptWeights), promptRows)] {
		weights[prefix+index] = weight
	}
	return weights
}
