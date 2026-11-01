package latentimage

import (
	"strings"
	"testing"
)

func TestLayoutPromptValidation(t *testing.T) {
	const layout = `{"summary":{"caption":"蓝色方块"},"items":[{"label":"object","caption":"cube","bbox":[-24,-16,24,16]}],"bbox_format":"yxyx","bbox_coord_space":"abs_centralized"}`
	for _, tc := range []struct {
		name          string
		mode          PromptMode
		prompt        string
		width, height int
		refusal       string
	}{
		{"text-is-opaque", promptText, "  not JSON\n", 32, 48, ""},
		{"omitted-is-opaque", "", `{incomplete`, 32, 48, ""},
		{"rectangle-edges", promptLayout, layout, 32, 48, ""},
		{"axis-order", promptLayout, layout, 48, 32, "y must be within [-16, 16]"},
		{"unknown-mode", "editing", layout, 32, 48, "unsupported"},
		{"malformed", promptLayout, `{incomplete`, 32, 48, "valid JSON"},
		{"missing-summary", promptLayout, strings.Replace(layout, `"caption":"蓝色方块"`, `"caption":" "`, 1), 32, 48, "summary.caption"},
		{"wrong-format", promptLayout, strings.Replace(layout, "yxyx", "xyxy", 1), 32, 48, "bbox_format"},
		{"wrong-space", promptLayout, strings.Replace(layout, "abs_centralized", "relative", 1), 32, 48, "bbox_coord_space"},
		{"missing-caption", promptLayout, strings.Replace(layout, `"caption":"cube"`, `"caption":""`, 1), 32, 48, "item 1"},
		{"missing-coordinate", promptLayout, strings.Replace(layout, "-24,-16,24,16", "-24,-16,24", 1), 32, 48, "four bbox"},
		{"inverted", promptLayout, strings.Replace(layout, "-24,-16,24,16", "24,-16,-24,16", 1), 32, 48, "y_min < y_max"},
		{"empty-area", promptLayout, strings.Replace(layout, "-24,-16,24,16", "0,-16,0,16", 1), 32, 48, "y_min < y_max"},
		{"outside-x", promptLayout, strings.Replace(layout, "-24,-16,24,16", "-24,-17,24,16", 1), 32, 48, "x within [-16, 16]"},
		{"null-coordinate", promptLayout, strings.Replace(layout, "-24,-16,24,16", "null,-16,24,16", 1), 32, 48, "coordinate"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validatePrompt(tc.mode, tc.prompt, tc.width, tc.height)
			if tc.refusal == "" && err != nil || tc.refusal != "" && (err == nil || !strings.Contains(err.Error(), tc.refusal)) {
				t.Fatal("layout result", err)
			}
		})
	}
}
