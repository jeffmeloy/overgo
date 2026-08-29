package routedlm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"overgo/internal/safetensors"
)

// PrefillValues: prompt input embeddings — token rows from the embedding
// table with image-mask positions replaced by projected image-feature rows
// (imageRow fills dst with the merged row for ordinal i).
func PrefillValues(src *safetensors.Source, cfg Config, b BranchBinding, inputIDs, imageMaskPositions []int, imageRows int, imageRow func(dst []float32, ordinal int) error) ([]float32, error) {
	if len(inputIDs) == 0 {
		return nil, fmt.Errorf("routed lm prefill: empty input ids")
	}
	if len(imageMaskPositions) != imageRows {
		return nil, fmt.Errorf("routed lm prefill: image mask positions=%d, want %d", len(imageMaskPositions), imageRows)
	}
	imageOrdinal := make([]int, len(inputIDs))
	for ordinal, pos := range imageMaskPositions {
		if pos < 0 || pos >= len(inputIDs) {
			return nil, fmt.Errorf("routed lm prefill: image mask position %d outside prompt len %d", pos, len(inputIDs))
		}
		if imageOrdinal[pos] != 0 {
			return nil, fmt.Errorf("routed lm prefill: duplicate image mask position %d", pos)
		}
		imageOrdinal[pos] = ordinal + 1
	}
	tokenIDs := make([]int, 0, len(inputIDs)-len(imageMaskPositions))
	tokenSlot := make(map[int]int, cap(tokenIDs))
	for pos, id := range inputIDs {
		if imageOrdinal[pos] != 0 {
			continue
		}
		if id < 0 || id >= cfg.VocabSize {
			return nil, fmt.Errorf("routed lm prefill: token id %d at pos %d outside vocab %d", id, pos, cfg.VocabSize)
		}
		if _, exists := tokenSlot[id]; !exists {
			tokenSlot[id] = len(tokenIDs)
			tokenIDs = append(tokenIDs, id)
		}
	}
	loadedRows, err := EmbeddingRows(src, cfg, b, tokenIDs)
	if err != nil {
		return nil, err
	}
	d := cfg.HiddenSize
	out := make([]float32, len(inputIDs)*d)
	for pos, id := range inputIDs {
		dst := out[pos*d : (pos+1)*d]
		if ordinal := imageOrdinal[pos]; ordinal != 0 {
			if err := imageRow(dst, ordinal-1); err != nil {
				return nil, err
			}
			continue
		}
		slot := tokenSlot[id]
		copy(dst, loadedRows[slot*d:(slot+1)*d])
	}
	return out, nil
}

// PromptSpecials: role-token ids resolved from tokenizer_config.json.
type PromptSpecials struct {
	BOS, EOS, User, Assistant                   int
	Think, ThinkEnd, Answer, AnswerEnd, NoThink int
	ImageStart, ImageEnd, Image, Video, NewLine int
	FlowLatent                                  int
}

// PromptRoleLiterals: role -> special-token literal. The caller supplies the
// checkpoint's role table (sourced from its tokenizer_config added tokens).
type PromptRoleLiterals struct {
	BOS, EOS, User, Assistant                   string
	Think, ThinkEnd, Answer, AnswerEnd, NoThink string
	ImageStart, ImageEnd, Image, Video, NewLine string
	FlowLatent                                  string
}

// LoadPromptSpecials: resolves each role literal to its id via
// tokenizer_config.json added_tokens_decoder.
func LoadPromptSpecials(modelDir string, roles PromptRoleLiterals) (PromptSpecials, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json"))
	if err != nil {
		return PromptSpecials{}, err
	}
	var parsed struct {
		AddedTokensDecoder map[string]struct {
			Content string `json:"content"`
		} `json:"added_tokens_decoder"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return PromptSpecials{}, fmt.Errorf("routed lm tokenizer_config: %w", err)
	}
	var out PromptSpecials
	bindings := []struct {
		role    string
		literal string
		dst     *int
	}{
		{"bos", roles.BOS, &out.BOS}, {"eos", roles.EOS, &out.EOS},
		{"user", roles.User, &out.User}, {"assistant", roles.Assistant, &out.Assistant},
		{"think", roles.Think, &out.Think}, {"think_end", roles.ThinkEnd, &out.ThinkEnd},
		{"answer", roles.Answer, &out.Answer}, {"answer_end", roles.AnswerEnd, &out.AnswerEnd},
		{"no_think", roles.NoThink, &out.NoThink},
		{"image_start", roles.ImageStart, &out.ImageStart}, {"image_end", roles.ImageEnd, &out.ImageEnd},
		{"image_context", roles.Image, &out.Image}, {"video_context", roles.Video, &out.Video},
		{"image_newline", roles.NewLine, &out.NewLine}, {"flow_latent", roles.FlowLatent, &out.FlowLatent},
	}
	wanted := make(map[string]*int, len(bindings))
	for _, b := range bindings {
		if b.literal == "" {
			return out, fmt.Errorf("routed lm tokenizer_config: missing role literal %s", b.role)
		}
		wanted[b.literal] = b.dst
	}
	for idText, token := range parsed.AddedTokensDecoder {
		id, err := strconv.Atoi(idText)
		if err != nil {
			return PromptSpecials{}, fmt.Errorf("routed lm tokenizer_config: invalid added token id %q", idText)
		}
		if dst := wanted[token.Content]; dst != nil {
			*dst = id
		}
	}
	for _, b := range bindings {
		if *b.dst <= 0 {
			return out, fmt.Errorf("routed lm tokenizer_config: missing special token %s (%q)", b.role, b.literal)
		}
	}
	return out, nil
}

// Encoder: text -> token ids (the prompt renderer's only tokenizer need).
type Encoder interface {
	Encode(text string) ([]int, error)
}

// RenderVisionQAPrompt: BOS, user, image block (rows of image-context tokens
// each closed by a newline token, then flow-latent + image-end), question,
// no-think, assistant, think, "\n\n", think-end, "\n", answer, "\n\n",
// answer-end. Ported verbatim from the reference prompt renderer.
func RenderVisionQAPrompt(tok Encoder, specials PromptSpecials, question string, gridH, gridW, merge int) ([]int, error) {
	if tok == nil {
		return nil, fmt.Errorf("routed lm vqa prompt: nil tokenizer")
	}
	if gridH <= 0 || gridW <= 0 || merge <= 0 || gridH%merge != 0 || gridW%merge != 0 {
		return nil, fmt.Errorf("routed lm vqa prompt: invalid grid h=%d w=%d merge=%d", gridH, gridW, merge)
	}
	ids := []int{specials.BOS, specials.User, specials.ImageStart}
	for range gridH / merge {
		for range gridW / merge {
			ids = append(ids, specials.Image)
		}
		ids = append(ids, specials.NewLine)
	}
	ids = append(ids, specials.FlowLatent, specials.ImageEnd)
	appendEncoded := func(text string) error {
		textIDs, err := tok.Encode(text)
		if err != nil {
			return fmt.Errorf("routed lm vqa prompt encode %q: %w", text, err)
		}
		ids = append(ids, textIDs...)
		return nil
	}
	if err := appendEncoded(question); err != nil {
		return nil, err
	}
	ids = append(ids, specials.NoThink, specials.Assistant, specials.Think)
	if err := appendEncoded("\n\n"); err != nil {
		return nil, err
	}
	ids = append(ids, specials.ThinkEnd)
	if err := appendEncoded("\n"); err != nil {
		return nil, err
	}
	ids = append(ids, specials.Answer)
	if err := appendEncoded("\n\n"); err != nil {
		return nil, err
	}
	ids = append(ids, specials.AnswerEnd)
	return ids, nil
}

// ImageMaskPositions: prompt positions holding image/video context tokens.
func ImageMaskPositions(ids []int, specials PromptSpecials) []int {
	var out []int
	for pos, id := range ids {
		if id == specials.Image || id == specials.Video {
			out = append(out, pos)
		}
	}
	return out
}

// ModalityMask: 1 at image positions, 0 elsewhere.
func ModalityMask(promptLen int, imageMaskPositions []int) []int {
	mask := make([]int, promptLen)
	for _, pos := range imageMaskPositions {
		if pos >= 0 && pos < promptLen {
			mask[pos] = 1
		}
	}
	return mask
}
