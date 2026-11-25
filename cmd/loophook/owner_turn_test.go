package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestAStopIsHeldToAnsweringARuleDecision holds doctrine-answer: while dispatch names a decision that changes a standing
// rule, a stop is blocked with the command that answers it; the operator's stop, a committing gate and a dispatch naming
// none let it pass. On 2026-11-23 a turn ended on such a decision and the plan sat 8 hours.
func TestAStopIsHeldToAnsweringARuleDecision(t *testing.T) {
	t.Parallel()
	waiting := plan.Dispatch{Waiting: "first is held by decision choose", Answer: []string{"choose"}}
	if hold := answerHold(waiting, false, false, false); !strings.Contains(hold, "go run ./cmd/plan -answer -title <option> choose") || !strings.Contains(hold, "the lead answers now") {
		t.Fatalf("a stop on an unanswered rule decision is held with %q; want the lead's command to answer it", hold)
	}
	for name, hold := range map[string]string{
		"the operator stopped the plan": answerHold(waiting, true, false, false),
		"a gate is committing":          answerHold(waiting, false, true, false),
		"no decision is the lead's":     answerHold(plan.Dispatch{Waiting: "first is held by decision push"}, false, false, false),
	} {
		if hold != "" {
			t.Errorf("%s: the stop is held with %q; want it to pass", name, hold)
		}
	}
}

// TestAStatusTurnEndsThoughADecisionWaits holds status-turn/do (SQA-490): with a decision that changes a standing rule
// waiting, a typed turn that leaves nothing uncommitted ends, the decision left for the next turn, while a turn no one
// typed, and a typed one that leaves work, is held to answer it. runStop had held every turn to the decision before it
// read the stop gate's verdict, so a typed question asking only for status was held.
func TestAStatusTurnEndsThoughADecisionWaits(t *testing.T) {
	t.Parallel()
	waiting := plan.Dispatch{Waiting: "first is held by decision choose", Answer: []string{"choose"}}
	if hold := answerHold(waiting, false, false, true); hold != "" || stopDecision(false, false, false, false, true) != stopAllow {
		t.Errorf("a clean typed turn with a decision waiting is held with %q; want it to end", hold)
	}
	if hold := answerHold(waiting, false, false, false); !strings.Contains(hold, "the lead answers now") {
		t.Errorf("an implementing turn with a decision waiting ended with %q; want it held to answer it", hold)
	}
}

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
