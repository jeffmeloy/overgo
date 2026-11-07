package main

import "testing"

// notificationPrompt is the opening of a background task's notification, a turn no person typed.
const notificationPrompt = "<task-notification><status>completed</status>"

// TestATurnTheOwnerStartedEndsWhenAnswered holds the Stop gate to the word typed on 2026-11-07: "answer the question
// and stop" was blocked twice, since a prompt answer did not pause the loop. A turn a typed message began ends once
// answered, the plan open and nothing uncommitted, with no stop recorded; one a task's notification or a scheduled wake
// began is held to the open row as before; and work the turn left uncommitted holds either turn.
func TestATurnTheOwnerStartedEndsWhenAnswered(t *testing.T) {
	t.Parallel()
	for prompt, typed := range map[string]bool{
		"answer the question and stop":                      true,
		"push to github when the readme table lands":        true,
		"<task-notification>\n<task-id>b42jttlpw</task-id>": false,
		"  <task-notification><status>completed</status>":   false,
		"<<autonomous-loop>>":                               false,
		"":                                                  false,
		"what does <task-notification> mean in the transcript, quoted?": true,
	} {
		if typedPrompt(prompt) != typed {
			t.Errorf("typedPrompt(%q) = %v, want %v", prompt, !typed, typed)
		}
	}
	for _, turn := range []struct {
		name             string
		dirty, typedTurn bool
		want             string
	}{
		{"a typed message's turn, answered", false, true, stopAllow},
		{"a notification's turn", false, false, stopContinue},
		{"a typed message's turn with work uncommitted", true, true, stopOrphan},
		{"a notification's turn with work uncommitted", true, false, stopOrphan},
	} {
		if got := stopDecision(false, false, false, turn.dirty, turn.typedTurn); got != turn.want {
			t.Errorf("%s: stopDecision = %q, want %q", turn.name, got, turn.want)
		}
	}
}
