package mediacapability

import (
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
)

// TestControlsDeclareTheWanPromptForm pins the page form of the Wan
// request: its prompts, seed and generation parameters are typed controls,
// its optional tensors and noise plan are omitted rather than refused, and
// LiveEdit's required condition still refuses until its request is
// assembled from a source clip.
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
	if _, refusal := Controls(modelrecipe.ModuleReferenceVideoPrepare, ""); !strings.Contains(refusal, "condition") {
		t.Fatalf("LiveEdit refusal = %q", refusal)
	}
}
