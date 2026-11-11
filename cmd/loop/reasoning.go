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

// stepSource is where a worker's step comes from: the plan's prompt for it and the step itself. It is the checkout's
// plan (liveSource) unless a test gives a fixed one, so cmd/loop's tests read no plan a landing changes, which ran them
// again at nearly every landing (fail-early/hermetic-tests).
type stepSource struct {
	prompt func() (string, error)
	step   func() (plan.Step, error)
}

// liveSource is the dispatched step of the checkout's plan, its prompt made by the plan command in root.
func liveSource(root string) stepSource {
	return stepSource{
		prompt: func() (string, error) { return runTool("go", "-C", root, "run", "./cmd/plan", "-prompt") },
		step:   currentStep,
	}
}

// reasoned is the step's prompt with its reasoning after it, or what failed to make either.
func (s stepSource) reasoned() (string, error) {
	prompt, err := s.prompt()
	if err != nil {
		return prompt, err
	}
	step, err := s.step()
	return withReasoning(prompt, step.Rationale), err
}
