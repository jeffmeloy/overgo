//go:build windows

package latentimage

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestARequestsLayoutBoxesBecomeItsPrompt holds mode 3 on a request (moss-ultra-sections/prompt-strings): its boxes
// make the prompt its text encoder reads on its canvas, as ComfyUI's MOSSImageLayoutPrompt makes it of the same boxes,
// no boxes leave the prompt as it is, a box without four finite edges is refused, and boxes are asked of a rotary
// profile alone.
func TestARequestsLayoutBoxesBecomeItsPrompt(t *testing.T) {
	boxes := &LayoutBoxes{Boxes: []LayoutBox{{Type: "object", Desc: "a table", Box: []float64{500, 100, 900, 400}}}}
	request := Request{Prompt: "a cafe at night", Width: 1024, Height: 576, Layout: boxes}
	// The golden's second layout (testdata/moss_prompt_strings/golden.json), ComfyUI's node on the same boxes.
	want := `{"summary":{"caption":"a cafe at night"},"items":[{"label":"object","caption":"a table","bbox":[0,-410,231,-102]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}`
	if got := request.prompt(); got != want {
		t.Errorf("the boxes made the prompt %s, ComfyUI's node %s", got, want)
	}
	testcheck.NoError(t, ValidateRequest(request))
	if got := (Request{Prompt: "a cafe", Layout: &LayoutBoxes{}}).prompt(); got != "a cafe" {
		t.Errorf("no boxes made the prompt %q, want it as it was", got)
	}
	request.Layout = &LayoutBoxes{Boxes: []LayoutBox{{Desc: "a table", Box: []float64{500, 100, 900}}}}
	if ValidateRequest(request) == nil {
		t.Error("a box of three edges was taken")
	}
	if !(Request{Layout: boxes}).rotaryOnly() {
		t.Error("layout boxes were offered to a flow profile")
	}
}
