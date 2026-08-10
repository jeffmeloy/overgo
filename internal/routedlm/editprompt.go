package routedlm

import (
	"fmt"
	"strings"

	"overgo/internal/hfbpe"
)

// SenseNova edit/generation prompt renderer. Ported verbatim from adaptive_new
// go/extmodel models.go compileCausalPrefixInput + compileCausalPrefixTemplate.
// Produces the token-id sequence and the per-row multi-axis positions (time,
// height, width) for one conditioning branch, with NO model forward required —
// the source image contributes only IMG_CONTEXT placeholder rows (soft tokens
// spliced in later), so the id/position sequence is fully determined by the
// text scaffold + source token geometry.

// PromptRole selects the conditioning branch, mirroring adaptive's
// GenerationConditionRole. Conditional == PromptSource; SourceOnly drops the
// system message and text prompt.
type PromptRole string

const (
	PromptRolePrompt        PromptRole = "prompt"        // text-only conditional
	PromptRoleUnconditional PromptRole = "unconditional" // empty user turn
	PromptRolePromptSource  PromptRole = "prompt_source" // text + source image (edit "conditional")
	PromptRoleSourceOnly    PromptRole = "source_only"   // source image, no text prompt
)

// PromptTemplate is the SenseNova-U1 MoT generation chat scaffold. Every field
// is a verbatim template fact recorded in the adaptive repodb "causal-gqa-image"
// reference-facts scope; provenance is cited per field. The renderer emits only
// these facts plus the source-image placeholder run — zero inline magic strings.
type PromptTemplate struct {
	SystemPrefix           string // "<|im_start|>system\n"  tokenizer_config.json chat_template.system
	SystemMessage          string // SYSTEM_MESSAGE_FOR_GEN  neo_unify/utils.py
	UserPrefix             string // "<|im_start|>user\n"    chat_template.user
	AssistantPrefix        string // "<|im_start|>assistant\n" chat_template.assistant
	Separator              string // "<|im_end|>\n"          chat_template.separator
	ImageStart             string // "<img>"                 added_tokens.json image_start
	ImageContext           string // "<IMG_CONTEXT>"         added_tokens.json image_context
	NonThinkingImagePrefix string // "<think>\n\n</think>\n\n<img>" neo_unify/utils.py non_thinking_image_prefix
}

// SenseNovaPromptTemplate returns the SenseNova-U1 template facts. These are the
// adaptive repodb "causal-gqa-image" scope values (reference_facts.json keys
// generation.template.* and generation.system_message); kept here as a cited
// contract function alongside SenseNovaBinding, since overgo has no model-fact
// store. The system message is the verbatim SYSTEM_MESSAGE_FOR_GEN string.
func SenseNovaPromptTemplate() PromptTemplate {
	return PromptTemplate{
		SystemPrefix: "<|im_start|>system\n",
		SystemMessage: "You are an image generation and editing assistant that accurately understands and executes " +
			"user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you " +
			"MUST start with a <think></think> block. Put all reasoning inside the block using plain text. " +
			"DO NOT include any image tags. Keep it reasonable and directly useful for producing the final " +
			"image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n" +
			"Task Types:\n\nA. Text-to-Image Generation:\n" +
			"- Generate a high-quality image based on the user's description.\n" +
			"- Ensure visual clarity, semantic consistency, and completeness.\n" +
			"- DO NOT introduce elements that contradict or override the user's intent.\n\n" +
			"B. Image Editing:\n" +
			"- Use the provided image(s) as input or reference for modification or transformation.\n" +
			"- The result can be an edited image or a new image based on the reference(s).\n" +
			"- Preserve all unspecified attributes unless explicitly changed.\n\n" +
			"General Rules:\n" +
			"- For any visible text in the image, follow the language specified for the rendered text in " +
			"the user's description, not the language of the prompt. If no language is specified, use the " +
			"user's input language.",
		UserPrefix:             "<|im_start|>user\n",
		AssistantPrefix:        "<|im_start|>assistant\n",
		Separator:              "<|im_end|>\n",
		ImageStart:             "<img>",
		ImageContext:           "<IMG_CONTEXT>",
		NonThinkingImagePrefix: "<think>\n\n</think>\n\n<img>",
	}
}

// PromptSource is the source-image token geometry (post-merge grid). TokenCount
// IMG_CONTEXT placeholders rasterize row-major over TokenWidth.
type PromptSource struct {
	TokenWidth  int
	TokenHeight int
	TokenCount  int
}

// EditPrompt is the rendered prefix input: ids plus per-row multi-axis
// positions. SourceRows lists the rows carrying IMG_CONTEXT placeholders.
type EditPrompt struct {
	IDs                 []int
	Time, Height, Width []int
	SourceRows          []int
	Role                PromptRole
}

// RenderEditPrompt builds the id sequence and (time,height,width) positions for
// one conditioning branch. tok is a byte-level BPE tokenizer over the model's
// vocab/merges (see hfbpe.LoadLegacy for SenseNova's split-file tokenizer).
func RenderEditPrompt(tok *hfbpe.Tokenizer, prompt string, role PromptRole, tmpl PromptTemplate, source PromptSource) (EditPrompt, error) {
	var out EditPrompt
	out.Role = role
	if tmpl.UserPrefix == "" || tmpl.AssistantPrefix == "" || tmpl.Separator == "" || tmpl.ImageStart == "" {
		return out, fmt.Errorf("edit prompt template is incomplete")
	}
	var query string
	switch role {
	case PromptRolePrompt:
		query = tmpl.SystemPrefix + tmpl.SystemMessage + tmpl.Separator + tmpl.UserPrefix + prompt + tmpl.Separator + tmpl.AssistantPrefix + tmpl.NonThinkingImagePrefix
	case PromptRoleUnconditional:
		query = tmpl.UserPrefix + tmpl.Separator + tmpl.AssistantPrefix + tmpl.ImageStart
	case PromptRolePromptSource:
		if source.TokenCount <= 0 || source.TokenWidth*source.TokenHeight != source.TokenCount || tmpl.ImageContext == "" {
			return out, fmt.Errorf("edit prefix has invalid source geometry or template")
		}
		if strings.Count(prompt, "<image>") > 1 {
			return out, fmt.Errorf("edit prefix supports one source image")
		}
		if !strings.Contains(prompt, "<image>") {
			prompt = "<image>\n" + prompt
		}
		imageTokens := tmpl.ImageStart + strings.Repeat(tmpl.ImageContext, source.TokenCount) + "</img>"
		prompt = strings.Replace(prompt, "<image>", imageTokens, 1)
		query = tmpl.SystemPrefix + tmpl.SystemMessage + tmpl.Separator + tmpl.UserPrefix + prompt + tmpl.Separator + tmpl.AssistantPrefix + tmpl.NonThinkingImagePrefix
	case PromptRoleSourceOnly:
		if source.TokenCount <= 0 || source.TokenWidth*source.TokenHeight != source.TokenCount || tmpl.ImageContext == "" {
			return out, fmt.Errorf("edit prefix has invalid source geometry or template")
		}
		imageTokens := tmpl.ImageStart + strings.Repeat(tmpl.ImageContext, source.TokenCount) + "</img>"
		query = tmpl.UserPrefix + imageTokens + tmpl.Separator + tmpl.AssistantPrefix + tmpl.ImageStart
	default:
		return out, fmt.Errorf("unsupported edit prompt role %q", role)
	}
	ids, err := tok.Encode(query)
	if err != nil {
		return EditPrompt{}, fmt.Errorf("encode edit prefix: %w", err)
	}
	out.IDs = ids
	out.Time = make([]int, len(ids))
	out.Height = make([]int, len(ids))
	out.Width = make([]int, len(ids))
	// Text-only branches advance time causally per row; no image geometry.
	if role == PromptRolePrompt || role == PromptRoleUnconditional {
		for row := range out.Time {
			out.Time[row] = row
		}
		return out, nil
	}
	imageContextID, ok := tok.SpecialID(tmpl.ImageContext)
	if !ok {
		return out, fmt.Errorf("image context token %q not in tokenizer", tmpl.ImageContext)
	}
	// Mask marks exactly the IMG_CONTEXT placeholder rows; BlockPositions then
	// reproduces the adaptive time/height/width accumulation (all placeholders
	// share one time index, rasterizing over the token grid).
	mask := make([]int, len(ids))
	for row, id := range ids {
		if id == imageContextID {
			mask[row] = 1
			out.SourceRows = append(out.SourceRows, row)
		}
	}
	if len(out.SourceRows) != source.TokenCount {
		return EditPrompt{}, fmt.Errorf("edit prefix has %d image rows, want %d", len(out.SourceRows), source.TokenCount)
	}
	positions, err := BlockPositions(mask, source.TokenWidth)
	if err != nil {
		return out, err
	}
	for row, pos := range positions {
		out.Time[row] = pos.Time
		out.Height[row] = pos.H
		out.Width[row] = pos.W
	}
	return out, nil
}
