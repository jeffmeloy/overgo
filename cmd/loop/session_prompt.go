package main

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"overgo/internal/jsonfile"
	"overgo/internal/plan"
)

// sessionInstructions are what a session the Claude app's scheduled task starts follows, printed by go run ./cmd/loop
// -session-prompt so the task's own text, one line, never changes and the instructions are versioned with the code.
// The session works the plan as the lead does; it never stops to ask; and one another session's hold sends away ends
// at once, so two runs never work the checkout together.
const sessionInstructions = `You are overgo's lead for this run, in the repository's main checkout. skill.md is the written guidance.

Work the plan, one step after another, until it is empty or a stop is recorded:
- go run ./cmd/plan -next names the step, and go run ./cmd/plan -prompt its check and notes.
- Make its check pass: go run ./cmd/plan -verify.
- Commit it: go run ./cmd/loop -land <item>/<step> -message-file <file>, run in the background, unpiped. Its message holds Cause:, Change:, Plan since the last landing, Checked by undoing, Findings pass and Predicted effect: lines, the prediction naming a scorecard metric and no measured figure.
- Then take the next step.

A command that refuses says what to do: do it and keep going. Never stop to ask; a choice between options takes the one that changes least and can be undone, recorded with go run ./cmd/plan -note. If the guard says another session holds this worktree, end this run at once: the session working it goes on.`

// writeSessionPrompt prints sessionInstructions and the environment every command of the session runs with: the
// driver's configuration at configPath names the data root and the role, and the session works as the lane's worker.
func writeSessionPrompt(configPath string, said io.Writer) error {
	var loaded config
	if err := jsonfile.DecodeStrict(configPath, &loaded); err != nil {
		return fmt.Errorf("load config: %w; write it with go run ./cmd/loop -init", err)
	}
	environment := []string{plan.AutomationWorkerEnvironment + "=" + loaded.Name}
	for name, value := range loaded.Environment {
		environment = append(environment, name+"="+value)
	}
	slices.Sort(environment)
	_, err := fmt.Fprintf(said, "%s\n\nRun every command with %s in its environment.\n", sessionInstructions, strings.Join(environment, " "))
	return err
}
