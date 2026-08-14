package inference

import (
	"testing"

	"overgo/internal/projector"
)

func TestProjectedInputsForPromptRejectsWidthMismatch(t *testing.T) {
	if _, _, err := ProjectedInputsForPrompt(&Runner{}, projector.MultimodalPrompt{EmbeddingWidth: 3}); err == nil {
		t.Fatal("projector/model width mismatch accepted")
	}
}
