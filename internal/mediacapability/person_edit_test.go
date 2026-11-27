package mediacapability

import (
	"encoding/json"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
)

// TestAnEditAtAPersonsSizeTakesASourceAtThatSize holds an edit's request at the size a person asks for to a source of
// that size: LiveEdit's verified request edits a 32 x 16 clip and states no size, its size being its source's, so its
// cell at a person's size measured nothing and its scorecard row kept no time there (2026-11-25, liveedit-size). Lifted,
// it edits the clip its entry declares at 832 x 480, its prompt and seed kept.
func TestAnEditAtAPersonsSizeTakesASourceAtThatSize(t *testing.T) {
	t.Parallel()
	verified := `{"prompt":"turn the stripes into falling snow","seed":1,"source_artifact":"file:sha256:465da1082505d34654fc678d45eb845b7d14f072718900d7b0089ceac6efecfb"}`
	sized, lifted, err := PersonRequest(modelrecipe.ModuleReferenceVideoPrepare, "", verified)
	testcheck.NoError(t, err)
	var request map[string]any
	testcheck.NoError(t, json.Unmarshal([]byte(sized), &request))
	want := "file:sha256:aba27ea7cdd3132034b1cdbbbf4c174dc80d02396af1afede152e4989f6e8bf2"
	if !lifted || request["source_artifact"] != want || request["prompt"] != "turn the stripes into falling snow" || request["seed"] != float64(1) {
		t.Fatalf("the edit at a person's size is %s (lifted %t); want the declared 832 x 480 source %s, its prompt and seed kept", sized, lifted, want)
	}
}
