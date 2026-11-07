package main

import (
	"io"
	"os"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// ruleWords are phrases of the rule text the hooks printed beside every message and at every block: "answer this
// request concisely, THEN continue the dispatched plan row", which contradicted "answer the question and stop" on
// 2026-11-07, "a prompt never pauses the loop", and "a turn ends only when the plan is EMPTY".
var ruleWords = []string{"BOUNDED REQUEST", "THEN continue", "never pauses", "does not pause", "ends only when", "TURN CONTRACT", "WORKTREE HELD"}

// TestHooksPrintNoRuleText holds the hooks to the step and its facts: the session-start text points to skill.md and
// states no rule of its own, the prompt hook prints no rule line for a question, a request or a session another holds,
// and the Stop gate's block names the open row without stating a rule.
func TestHooksPrintNoRuleText(t *testing.T) {
	// Serial: the prompt hook reads and writes the checkout's process state, and the test captures standard output.
	write := sessionCheckout(t)
	write("work.txt", "a file that was there\n")
	ruled := func(text string) string {
		for _, words := range ruleWords {
			if strings.Contains(text, words) {
				return words
			}
		}
		return ""
	}
	if words := ruled(doctrineText); words != "" || !strings.Contains(doctrineText, "skill.md") || strings.Contains(doctrineText, "(check:") {
		t.Fatalf("the session-start text states a rule (%q) or names no skill.md: %s", words, doctrineText)
	}
	prompted := func(session, prompt string) string {
		t.Helper()
		read, written, err := os.Pipe()
		testcheck.NoError(t, err)
		saved := os.Stdout
		os.Stdout = written
		runPrompt(`{"session_id": "` + session + `", "prompt": "` + prompt + `"}`)
		os.Stdout = saved
		testcheck.NoError(t, written.Close())
		return string(testcheck.Must(io.ReadAll(read))(t))
	}
	for _, turn := range []struct{ session, prompt string }{
		{"lead", "what is going on with the stoppages?"},
		{"lead", "summarize the plan"},
		{"lead", "continue with the plan"},
		{"second", "work the row as well"},
	} {
		if printed := prompted(turn.session, turn.prompt); ruled(printed) != "" {
			t.Errorf("the prompt hook printed rule text (%q) for %q: %s", ruled(printed), turn.prompt, printed)
		}
	}
	for _, progressed := range []bool{false, true} {
		block := stopMessage(stopContinue, progressed, plan.Dispatch{Line: "row / step: the work"})
		if words := ruled(block); words != "" || !strings.Contains(block, "row / step") {
			t.Errorf("the Stop gate's block states a rule (%q) or names no row: %s", words, block)
		}
	}
}
