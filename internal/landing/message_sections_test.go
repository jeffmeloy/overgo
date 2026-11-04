package landing

import (
	"strings"
	"testing"
)

// TestAMessageWithoutItsSectionsIsRefused holds a landing's message to its two sections: one that lacks the cause or
// the predicted effect is refused naming the missing one, and one that opens a line with each passes.
func TestAMessageWithoutItsSectionsIsRefused(t *testing.T) {
	t.Parallel()
	for message, missing := range map[string]string{
		"row/do: change\n\nPredicted effect: first-output holds.\n": "Cause:",
		"row/do: change\n\nCause: a fixture.\n":                     predictedSection,
		"row/do: change\n\nThe cause and the effect, in prose.\n":   "Cause:",
	} {
		if err := requireSections([]byte(message)); err == nil || !strings.Contains(err.Error(), missing) {
			t.Errorf("a message lacking %q ended as %v, want it refused naming that section", missing, err)
		}
	}
	if err := requireSections([]byte("row/do: change\n\nCause: a fixture.\n" + predictedSection + " first-output holds.\n")); err != nil {
		t.Errorf("a message with both sections was refused: %v", err)
	}
}
