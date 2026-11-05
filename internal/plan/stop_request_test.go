package plan

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestARequestedStopNeedsARequest holds a requested stop to the latest prompt in the worktree: it is refused until a
// prompt asks to stop, by a whole word read without case or punctuation, and again once a later prompt does not.
func TestARequestedStopNeedsARequest(t *testing.T) {
	t.Parallel()
	for prompt, asks := range map[string]bool{
		"please stop the loop":     true,
		"Pause.":                   true,
		"HALT now":                 true,
		"keep working":             false,
		"the loop is unstoppable":  false,
		"check the stopwatch code": false,
	} {
		if asksToStop(prompt) != asks {
			t.Errorf("asksToStop(%q) = %v, want %v", prompt, !asks, asks)
		}
	}
	root := t.TempDir()
	if err := RequireStopRequest(root); err == nil {
		t.Fatal("a requested stop was admitted with no prompt asking for one")
	}
	for _, prompt := range []string{"stop here, I will look at the run", "keep working on the next row"} {
		testcheck.NoError(t, NoteStopRequest(root, prompt))
		if err := RequireStopRequest(root); (err == nil) != asksToStop(prompt) {
			t.Fatalf("after the prompt %q a requested stop was admitted=%v, want %v", prompt, err == nil, asksToStop(prompt))
		}
	}
}
