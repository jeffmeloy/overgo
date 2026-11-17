//go:build windows

package latentimage

import (
	"math"
	"testing"

	"overgo/internal/testcheck"
)

// TestARequestsLayoutBecomesItsPrompt holds mode 3 on a request (moss-ultra-sections/mode-three): its layout makes the
// prompt its text encoder reads on its canvas, as ComfyUI's nodes make it of the same fields, in place of a prompt of
// its own, which beside it is refused; a style the builder does not offer and a region of no finite extent are refused;
// and a layout is asked of a rotary profile alone.
func TestARequestsLayoutBecomesItsPrompt(t *testing.T) {
	draft := &LayoutDraft{Background: "a cobbled street", Style: draftStylePhoto, Photo: "35mm",
		Regions: []LayoutRegion{{X: 0.1, Y: 0.2, W: 0.35, H: 0.4, Desc: "a café table"}}}
	request := Request{Width: 1024, Height: 576, Layout: draft}
	testcheck.NoError(t, ValidateRequest(request))
	// Create Bounding Boxes rounds the region's corner and width, 102.4 and 358.4 pixels, to 102 and 358, so its right
	// edge, 450 on the builder's grid, is 449 on its own and 52 pixels left of the centre as the layout ceils it.
	want := `{"summary":{"caption":"Scene: \nBackground: a cobbled street\nAesthetics: \nLighting: \nCamera: 35mm\nMedium: \nArt style: \nPalette:"},` +
		`"items":[{"label":"object","caption":"a café table","bbox":[-173,-410,58,-52]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}`
	if got := request.prompt(); got != want {
		t.Errorf("the layout made the prompt %s, want %s", got, want)
	}
	if got := (Request{Prompt: "a cafe", Width: 1024, Height: 576}).prompt(); got != "a cafe" {
		t.Errorf("a request without a layout reads %q, want its prompt", got)
	}
	for name, refused := range map[string]Request{
		"a prompt beside it": {Prompt: "a cafe", Width: 1024, Height: 576, Layout: draft},
		"an unknown style":   {Width: 1024, Height: 576, Layout: &LayoutDraft{Style: "cartoon"}},
		"an endless region":  {Width: 1024, Height: 576, Layout: &LayoutDraft{Regions: []LayoutRegion{{W: math.Inf(1)}}}},
	} {
		if ValidateRequest(refused) == nil {
			t.Errorf("a layout with %s was taken", name)
		}
	}
	if !(Request{Layout: draft}).rotaryOnly() {
		t.Error("a layout was offered to a flow profile")
	}
}
