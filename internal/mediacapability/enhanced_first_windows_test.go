//go:build windows

package mediacapability

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
)

// TestAPictureRequestAskingNoEnhancingGoesToItsSessionAsItIs holds the image capability's prompt mode 2 step
// (moss-ultra-sections/prompt-modes): a request asking no enhancing, and an input that is no request, reach the
// picture session as they came, the session's answer returned, no chat opened.
func TestAPictureRequestAskingNoEnhancingGoesToItsSessionAsItIs(t *testing.T) {
	var reached []string
	session := func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		reached = append(reached, input)
		return "drawn", nil
	}
	inputs := []string{`{"prompt":"a fox","width":1024,"height":576}`, `{"prompt":"a fox","enhance":false}`, `not a request`}
	for _, input := range inputs {
		if got := testcheck.Must(enhancedFirst(session)(t.Context(), nil, "", modelrecipe.CapabilityEvidenceSelection{}, input))(t); got != "drawn" {
			t.Errorf("%s gave %v, want the session's answer", input, got)
		}
	}
	for index, input := range inputs {
		if reached[index] != input {
			t.Errorf("the session took %q, want %q as it came", reached[index], input)
		}
	}
}
