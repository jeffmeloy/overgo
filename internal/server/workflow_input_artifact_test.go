package server

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestWorkflowInputArtifactControl pins the artifact control's validation:
// the page fills it from a media card's stored id, and a value that is no
// artifact id is refused before any run, while a well-formed id passes.
func TestWorkflowInputArtifactControl(t *testing.T) {
	if !WorkflowControlArtifact.valid() {
		t.Fatal("the artifact control type is not a valid capability control")
	}
	controls := []WorkflowControl{{Name: "source_artifact", Type: WorkflowControlArtifact}, {Name: "seed", Type: WorkflowControlInteger}}
	valid := json.RawMessage(`{"source_artifact": "file:sha256:` + strings.Repeat("ab", 32) + `", "seed": 7}`)
	if err := validateWorkflowInput(controls, valid); err != nil {
		t.Fatal(err)
	}
	for name, raw := range map[string]json.RawMessage{
		"not an id":    json.RawMessage(`{"source_artifact": "clip.gif"}`),
		"not a string": json.RawMessage(`{"source_artifact": 7}`),
	} {
		if err := validateWorkflowInput(controls, raw); err == nil {
			t.Errorf("%s: admitted %s", name, raw)
		}
	}
}
