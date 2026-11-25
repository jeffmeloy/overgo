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

// TestAStopRequestIsTheAskingSessions holds stop-request-session/do's word rule (SQA-487): a stop word the word before
// it negates asks for no stop. "don't stop", "do not pause", "never halt" and "won't stop" are no request, kept or
// noted, while "stop now" and "please pause" are. Its session rule is held by the test of this name in cmd/loophook.
func TestAStopRequestIsTheAskingSessions(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	for prompt, asks := range map[string]bool{
		"don't stop, keep going": false, "do not pause the loop": false, "never halt": false, "it won't stop": false,
		"stop now": true, "please pause": true,
	} {
		testcheck.NoError(t, NoteStopRequest(root, prompt))
		if err := RequireStopRequest(root); (err == nil) != asks {
			t.Errorf("after the prompt %q a requested stop was admitted=%t; want %t", prompt, err == nil, asks)
		}
	}
}
