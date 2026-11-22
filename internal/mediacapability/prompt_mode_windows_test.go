//go:build windows

package mediacapability

import (
	"encoding/json"
	"maps"
	"slices"
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestMOSSOffersOnePromptModeShowingItsFields holds moss-prompt-mode/controls: the MOSS Ultra workflow chooses its
// prompt one way (PROMPT MODE 483: a prompt as written, the enhancer's from an idea, or the builder's from a layout),
// and MOSS's control list offers that one choice, prompt_mode, its choices text and layout (a prompt as written, the
// one checked as MOSS's layout JSON), enhanced and builder, text's fields where none is chosen and no default written into a
// request (a request recorded with one would answer from no stored run), each declaring the fields it shows
// (Control.Shows): the prompt and negative for the first three, the layout and negative for the builder; every field
// named is a control of the list, and no control asks mode 2 apart from it. A request a mode's fields make is taken as
// that mode: enhanced is enhanced before its session (latentimage Request.Enhanced), the builder's layout makes its
// prompt, and a builder with no layout, a builder beside a prompt and a layout in another mode are refused. The page had
// offered prompt_mode (a check, text or layout), enhance and layout side by side, so it could not show one mode's fields
// alone.
func TestMOSSOffersOnePromptModeShowingItsFields(t *testing.T) {
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	controls, refusal := Controls(modelrecipe.ModuleLatentImagePrepare, root)
	if refusal != "" {
		t.Fatal(refusal)
	}
	names := map[string]bool{}
	var mode Control
	for _, control := range controls {
		names[control.Name] = true
		if control.Name == "prompt_mode" {
			mode = control
		}
	}
	written := []string{"prompt", "negative_prompt"}
	want := map[string][]string{"": written, "text": written, "layout": written, "enhanced": written, "builder": {"layout", "negative_prompt"}}
	if !slices.Equal(mode.Choices, []string{"text", "layout", "enhanced", "builder"}) || mode.Default != "" ||
		!maps.EqualFunc(mode.ChoiceShows(), want, slices.Equal[[]string]) {
		t.Fatalf("MOSS's prompt mode offers %v (default %q) showing %v; want text, layout, enhanced and builder, none filled where none is chosen, showing %v", mode.Choices, mode.Default, mode.Shows, want)
	}
	for _, shown := range mode.Shows {
		for _, name := range shown {
			if !names[name] {
				t.Errorf("a prompt mode shows %s, which MOSS's controls do not offer", name)
			}
		}
	}
	if names["enhance"] {
		t.Error("MOSS's controls offer enhance beside its prompt mode")
	}

	taken := func(raw string) (latentimage.Request, error) {
		t.Helper()
		var request latentimage.Request
		testcheck.NoError(t, strictjson.DecodeBytes([]byte(raw), &request))
		return request, latentimage.Admit(request)
	}
	const sized = `"width":1024,"height":576`
	layout := `"layout":{"high_level_description":"a fox at dawn","regions":[{"x":0.1,"y":0.2,"w":0.4,"h":0.5,"desc":"a fox"}]}`
	for _, admitted := range []string{
		`{"prompt_mode":"text","prompt":"a fox","negative_prompt":"blur",` + sized + `}`,
		`{"prompt_mode":"enhanced","prompt":"a fox","negative_prompt":"blur",` + sized + `}`,
		`{"prompt_mode":"builder",` + layout + `,"negative_prompt":"blur",` + sized + `}`,
	} {
		request, err := taken(admitted)
		if err != nil {
			t.Errorf("%s was refused: %v", admitted, err)
		}
		if enhanced := request.PromptMode == "enhanced"; request.Enhanced() != enhanced {
			t.Errorf("%s is taken as enhanced %t", admitted, request.Enhanced())
		}
	}
	for _, refused := range []string{
		`{"prompt_mode":"builder","negative_prompt":"blur",` + sized + `}`,
		`{"prompt_mode":"builder","prompt":"a fox",` + layout + `,` + sized + `}`,
		`{"prompt_mode":"text",` + layout + `,` + sized + `}`,
	} {
		if _, err := taken(refused); err == nil {
			t.Errorf("%s was admitted", refused)
		}
	}
	raw := testcheck.Must(json.Marshal(mode.Shows))(t)
	t.Logf("MOSS's prompt modes show %s", raw)
}
