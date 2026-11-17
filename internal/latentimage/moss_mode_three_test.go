package latentimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/testcheck"
)

// TestModeThreeEqualsComfyUIsNodes holds prompt mode 3 (moss-ultra-sections/mode-three) to ComfyUI's own nodes, run on
// fixed builder fields and regions by scripts/python_golden_references/comfy_mode_three_golden.py: the eight lines the
// extracts and StringFormat make of the builder's caption, the elements Create Bounding Boxes gives back on the canvas,
// and the layout prompt MOSSImageLayoutPrompt makes of both, each equal to theirs; the case the recording drew, every
// field empty and no region, is its eight empty lines.
func TestModeThreeEqualsComfyUIsNodes(t *testing.T) {
	var golden struct {
		Cases []struct {
			Inputs struct {
				Name       string            `json:"name"`
				Width      int               `json:"width"`
				Height     int               `json:"height"`
				Style      string            `json:"style"`
				Fields     map[string]string `json:"fields"`
				Palette    []string          `json:"palette"`
				Regions    []LayoutRegion    `json:"regions"`
				Characters string            `json:"characters"`
			} `json:"inputs"`
			Prompt   string          `json:"prompt"`
			Elements []layoutElement `json:"elements"`
			Layout   string          `json:"layout"`
		} `json:"cases"`
	}
	raw := testcheck.Must(os.ReadFile(filepath.Join("testdata", "moss_mode_three", "golden.json")))(t)
	testcheck.NoError(t, json.Unmarshal(raw, &golden))
	if len(golden.Cases) == 0 {
		t.Fatal("the golden holds no case")
	}
	for _, recorded := range golden.Cases {
		in := recorded.Inputs
		draft := LayoutDraft{
			Description: in.Fields["high_level_description"], Background: in.Fields["background"], Style: in.Style,
			Photo: in.Fields["photo"], ArtStyle: in.Fields["art_style"], Aesthetics: in.Fields["aesthetics"],
			Lighting: in.Fields["lighting"], Medium: in.Fields["medium"], Palette: in.Palette, Regions: in.Regions,
			Characters: in.Characters,
		}
		testcheck.NoError(t, draft.validate())
		if got := draft.text(); got != recorded.Prompt {
			t.Errorf("%s: the fields read %q, ComfyUI's %q", in.Name, got, recorded.Prompt)
		}
		elements := make([]layoutElement, len(draft.Regions))
		for index, region := range draft.Regions {
			elements[index] = region.element(in.Width, in.Height)
		}
		if len(elements) != 0 || len(recorded.Elements) != 0 {
			if !reflect.DeepEqual(elements, recorded.Elements) {
				t.Errorf("%s: the regions come back as %+v, from Create Bounding Boxes as %+v", in.Name, elements, recorded.Elements)
			}
		}
		if got := draft.prompt(in.Width, in.Height); got != recorded.Layout {
			t.Errorf("%s: the layout is %s, ComfyUI's %s", in.Name, got, recorded.Layout)
		}
	}
}
