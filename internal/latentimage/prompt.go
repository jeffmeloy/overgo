package latentimage

import (
	"encoding/json"
	"fmt"
	"strings"
)

// PromptMode is how a request gives the prompt its picture is drawn from, the MOSS Ultra workflow's prompt mode: text,
// a prompt as written (mode 1), or layout, a prompt written as MOSS's layout JSON, checked against its canvas before it
// is drawn; enhanced, a source idea and exclusions of which the model's enhancer chat writes the prompt and negative
// (mode 2, EnhanceRequest); builder, the Ideogram 4 builder's fields, which make the layout prompt (mode 3, Layout). A
// request had asked mode 2 by a flag beside the mode (enhance) and mode 3 by its layout alone, so a page offered three
// controls for the workflow's one choice (moss-prompt-mode).
type PromptMode string

const (
	// promptText preserves free-form text, including serialized prompts.
	promptText PromptMode = "text"
	// promptLayout validates the centered yxyx JSON format documented by MOSS-Image.
	promptLayout PromptMode = "layout"
	// promptEnhanced is prompt mode 2: the enhancer chat writes the prompt and negative of the request's idea.
	promptEnhanced PromptMode = "enhanced"
	// promptBuilder is prompt mode 3: the request's layout makes its prompt.
	promptBuilder PromptMode = "builder"
	// layoutAxes counts the horizontal and vertical axes of centered yxyx boxes.
	layoutAxes = 2
)

// Centered yxyx positions follow the source layout format's corner ordering.
const (
	layoutYMin = iota
	layoutXMin
	layoutYMax
	layoutXMax
	layoutCoordinates
)

// Choices lists the prompt modes.
func (PromptMode) Choices() []string {
	return []string{string(promptText), string(promptLayout), string(promptEnhanced), string(promptBuilder)}
}

// Shows are the request's fields each prompt mode gives its prompt by, the rest of a mode's fields the request's own, a
// request naming no mode taking text's (validatePrompt):
// text and layout a prompt and negative, enhanced an idea and exclusions in the same fields, builder its layout and a
// negative (moss-prompt-mode).
func (PromptMode) Shows() map[string][]string {
	written := []string{"prompt", "negative_prompt"}
	return map[string][]string{
		"": written, string(promptText): written, string(promptLayout): written, string(promptEnhanced): written,
		string(promptBuilder): {"layout", "negative_prompt"},
	}
}

// Help describes the prompt modes and the source layout convention, with a plain prompt example.
func (PromptMode) Help() string {
	return `Text accepts any prompt. Layout checks JSON coordinates before generation. Enhanced writes the prompt and negative from the idea and exclusions given (MOSS Ultra's prompt mode 2). Builder makes the prompt from the layout's fields (prompt mode 3). Boxes use [y_min, x_min, y_max, x_max] in pixels from the canvas center. A 1024 by 1024 canvas spans -512 to 512 on each axis. Example: {"summary":{"caption":"A blue cube"},"items":[{"label":"object","caption":"blue cube","bbox":[-128,-128,128,128]}],"bbox_format":"yxyx","bbox_coord_space":"abs_centralized"}`
}

// validatePrompt checks an explicitly selected layout against its canvas.
// Text mode leaves arbitrary text and JSON prompts unchanged; enhanced and builder are checked where their prompt is
// written (EnhanceRequest) or made (ValidateRequest's layout).
func validatePrompt(mode PromptMode, prompt string, width, height int) error {
	switch mode {
	case "", promptText, promptEnhanced, promptBuilder:
		return nil
	case promptLayout:
	default:
		return fmt.Errorf("latent image: prompt mode %q is unsupported; use text, layout, enhanced or builder", mode)
	}
	var layout struct {
		Summary struct {
			Caption string `json:"caption"`
		} `json:"summary"`
		Items []struct {
			Label   string     `json:"label"`
			Caption string     `json:"caption"`
			BBox    []*float64 `json:"bbox"`
		} `json:"items"`
		Format string `json:"bbox_format"`
		Space  string `json:"bbox_coord_space"`
	}
	if err := json.Unmarshal([]byte(prompt), &layout); err != nil {
		return fmt.Errorf("latent image: layout prompt must be valid JSON: %w", err)
	}
	if strings.TrimSpace(layout.Summary.Caption) == "" || len(layout.Items) == 0 {
		return fmt.Errorf("latent image: layout needs summary.caption and at least one item")
	}
	if layout.Format != "yxyx" || layout.Space != "abs_centralized" {
		return fmt.Errorf("latent image: layout requires bbox_format yxyx and bbox_coord_space abs_centralized")
	}
	// Centered bounds divide each canvas axis into its two equal halves.
	halfWidth, halfHeight := float64(width)/layoutAxes, float64(height)/layoutAxes
	for i, item := range layout.Items {
		if strings.TrimSpace(item.Label) == "" || strings.TrimSpace(item.Caption) == "" {
			return fmt.Errorf("latent image: layout item %d needs label and caption", i+1)
		}
		// yxyx carries two corners with two coordinates each.
		if len(item.BBox) != layoutCoordinates {
			return fmt.Errorf("latent image: layout item %d needs four bbox coordinates [y_min, x_min, y_max, x_max]", i+1)
		}
		var box [layoutCoordinates]float64
		for j, coordinate := range item.BBox {
			if coordinate == nil {
				return fmt.Errorf("latent image: layout item %d bbox coordinate %d must be a number", i+1, j+1)
			}
			box[j] = *coordinate
		}
		if !(box[layoutYMin] < box[layoutYMax] && box[layoutXMin] < box[layoutXMax]) {
			return fmt.Errorf("latent image: layout item %d bbox must have y_min < y_max and x_min < x_max", i+1)
		}
		if box[layoutYMin] < -halfHeight || box[layoutYMax] > halfHeight || box[layoutXMin] < -halfWidth || box[layoutXMax] > halfWidth {
			return fmt.Errorf("latent image: layout item %d bbox exceeds the canvas; y must be within [%g, %g] and x within [%g, %g]", i+1, -halfHeight, halfHeight, -halfWidth, halfWidth)
		}
	}
	return nil
}
