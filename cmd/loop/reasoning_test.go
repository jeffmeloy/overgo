package main

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/loop"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestTheLocalWorkerIsHandedTheStepsReasoning holds the driver's prompt and show_step to the step's reasoning
// (loop-proof/live-reasoning): the plan's prompt shows a step's notes and none of the rest, and the local worker reads
// only what it is handed, so a live step reached it without the reasoning its replays were handed, though show_step said
// it showed it. Each holds the step's reasoning that is not a note, and neither repeats a note as a paragraph, the prompt
// showing each as a NOTE line already. The step is a fixed one, so the test reads no plan a landing changes
// (fail-early/hermetic-tests).
func TestTheLocalWorkerIsHandedTheStepsReasoning(t *testing.T) {
	t.Parallel()
	step := plan.Step{ID: "step", Title: "A step", Rationale: "Serves: the owner.\n\nDisappears: a gap.\n\nNote: run 1 passed.\n\nCheck: TestIt."}
	fixed := stepSource{
		prompt: func() (string, error) { return "TASK row/step\nA step\nNOTE run 1 passed.\n", nil },
		step:   func() (plan.Step, error) { return step, nil },
	}
	prompt := testcheck.Must((&execWorld{steps: fixed}).Prompt(loop.Step{Item: "row", ID: "step"}))(t)
	tools := workerTools(t.TempDir(), planStepFrom(t.TempDir(), fixed))
	shown := testcheck.Must(tools[slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == "show_step" })].run(nil))(t)
	for name, text := range map[string]string{"the driver's prompt": prompt, "show_step": shown} {
		for _, paragraph := range []string{"Serves: the owner.", "Disappears: a gap.", "Check: TestIt."} {
			if !strings.Contains(text, paragraph) {
				t.Errorf("%s holds no paragraph %q of the step's reasoning:\n%s", name, paragraph, text)
			}
		}
		if strings.Contains(text, "\n\nNote: ") {
			t.Errorf("%s repeats a note as a paragraph:\n%s", name, text)
		}
	}
}
