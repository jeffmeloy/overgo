package mediacapability

import (
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
)

// TestControlsDeclareTheWanPromptForm pins the page forms of the video
// requests: Wan's prompts, seed and generation parameters are typed
// controls with its optional tensors and noise plan omitted rather than
// refused, and LiveEdit declares its prompt, seed and source clip artifact
// with its compiled condition and decoded source omitted. Each offers the
// formats its clip is written in as an optional choice, unset for a GIF.
func TestControlsDeclareTheWanPromptForm(t *testing.T) {
	controls, refusal := Controls(modelrecipe.ModuleLatentVideoPrepare, "")
	if refusal != "" {
		t.Fatal(refusal)
	}
	byName := map[string]Control{}
	for _, control := range controls {
		byName[control.Name] = control
	}
	for name, kind := range map[string]string{
		"prompt": ControlText, "negative_prompt": ControlText, "seed": ControlInteger,
		"frames": ControlInteger, "width": ControlInteger, "height": ControlInteger, "steps": ControlInteger,
		"shift": ControlNumber, "guide_scale": ControlNumber,
	} {
		if byName[name].Type != kind {
			t.Errorf("control %s = %+v want type %s", name, byName[name], kind)
		}
	}
	for _, name := range []string{"cond_context", "uncond_context", "initial_sample", "noise"} {
		if _, listed := byName[name]; listed {
			t.Errorf("optional tensor field %s is listed as a control", name)
		}
	}
	edit, refusal := Controls(modelrecipe.ModuleReferenceVideoPrepare, "")
	if refusal != "" {
		t.Fatalf("LiveEdit refusal = %q", refusal)
	}
	names := make([]string, 0, len(edit))
	for _, control := range edit {
		names = append(names, control.Name+":"+control.Type)
	}
	if joined := strings.Join(names, ","); joined != "prompt:text,seed:integer,source_artifact:artifact,format:text" {
		t.Fatalf("LiveEdit controls = %s", joined)
	}
	for request, format := range map[string]Control{"Wan": byName["format"], "LiveEdit": edit[len(edit)-1]} {
		if format.Required || strings.Join(format.Choices, ",") != "gif,mp4" {
			t.Errorf("%s format control = %+v, want an optional choice of gif and mp4", request, format)
		}
	}
}
