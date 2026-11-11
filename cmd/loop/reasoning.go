package main

import (
	"strings"

	"overgo/internal/plan"
)

// reasoningBreak parts a step's reasoning into paragraphs, as plan -note writes them.
const reasoningBreak = "\n\n"

// withReasoning is a step's prompt with the step's reasoning after it, its notes left out, which the prompt already
// shows. The plan's prompt shows a step's notes and none of the rest: a session reads the plan itself, and the local
// worker, which reads only what it is handed, worked a live step without the reasoning its replays were handed
// (replayed.prompt), though show_step said it showed it (loop-proof/live-reasoning).
func withReasoning(prompt, rationale string) string {
	var kept []string
	for paragraph := range strings.SplitSeq(rationale, reasoningBreak) {
		if strings.TrimSpace(paragraph) != "" && !strings.HasPrefix(paragraph, plan.NotePrefix) {
			kept = append(kept, paragraph)
		}
	}
	if len(kept) == 0 {
		return prompt
	}
	return strings.TrimRight(prompt, "\n") + reasoningBreak + "REASONING\n" + strings.Join(kept, reasoningBreak) + "\n"
}

// reasonedPrompt is the plan's prompt with the dispatched step's reasoning after it, or what failed to make either.
func reasonedPrompt(prompt string, err error) (string, error) {
	if err != nil {
		return prompt, err
	}
	step, err := currentStep()
	return withReasoning(prompt, step.Rationale), err
}
