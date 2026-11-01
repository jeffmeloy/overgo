package latentimage

import (
	"encoding/json"
	"fmt"
	"strings"
)

// PromptMode selects optional validation without rewriting the prompt bytes.
type PromptMode string

const (
	// promptText preserves free-form text, including serialized prompts.
	promptText PromptMode = "text"
	// promptLayout validates the centered yxyx JSON format documented by MOSS-Image.
	promptLayout PromptMode = "layout"
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

// Choices lists the supported prompt validation modes.
func (PromptMode) Choices() []string { return []string{string(promptText), string(promptLayout)} }

// Help describes the source layout convention and supplies a plain prompt example.
func (PromptMode) Help() string {
	return `Text accepts any prompt. Layout checks JSON coordinates before generation. Boxes use [y_min, x_min, y_max, x_max] in pixels from the canvas center. A 1024 by 1024 canvas spans -512 to 512 on each axis. Example: {"summary":{"caption":"A blue cube"},"items":[{"label":"object","caption":"blue cube","bbox":[-128,-128,128,128]}],"bbox_format":"yxyx","bbox_coord_space":"abs_centralized"}`
}

// validatePrompt checks an explicitly selected layout against its canvas.
// Text mode leaves arbitrary text and JSON prompts unchanged.
func validatePrompt(mode PromptMode, prompt string, width, height int) error {
	switch mode {
	case "", promptText:
		return nil
	case promptLayout:
	default:
		return fmt.Errorf("latent image: prompt mode %q is unsupported; use text or layout", mode)
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
