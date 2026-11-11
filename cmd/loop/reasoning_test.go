package main

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/loop"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// TestTheLocalWorkerIsHandedTheStepsReasoning holds the driver's prompt and show_step to the dispatched step's reasoning
// (loop-proof/live-reasoning): the plan's prompt shows a step's notes and none of the rest, and the local worker reads
// only what it is handed, so a live step reached it without the reasoning its replays were handed, though show_step said
// it showed it. Each holds the step's first paragraph of reasoning that is not a note, and neither repeats a note as a
// paragraph, the prompt showing each as a NOTE line already.
func TestTheLocalWorkerIsHandedTheStepsReasoning(t *testing.T) {
	root := testutil.RepoRoot(t)
	t.Chdir(root)
	dispatch := testcheck.Must(dispatched())(t)
	step := testcheck.Must(dispatchedStep(dispatch))(t)
	var paragraph string
	for candidate := range strings.SplitSeq(step.Rationale, "\n\n") {
		if strings.TrimSpace(candidate) != "" && !strings.HasPrefix(candidate, "Note: ") {
			paragraph = candidate
			break
		}
	}
	if paragraph == "" {
		testskip.NotApplicable(t, "the dispatched step has no reasoning beside its notes")
	}
	prompt := testcheck.Must((&execWorld{}).Prompt(loop.Step{Item: dispatch.Item, ID: dispatch.Step}))(t)
	tools := workerTools(root, planStep(root))
	shown := testcheck.Must(tools[slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == "show_step" })].run(nil))(t)
	for name, text := range map[string]string{"the driver's prompt": prompt, "show_step": shown} {
		if !strings.Contains(text, paragraph) {
			t.Errorf("%s for %s/%s holds no paragraph %q of its reasoning:\n%s", name, dispatch.Item, dispatch.Step, paragraph, text)
		}
		if strings.Contains(text, "\n\nNote: ") {
			t.Errorf("%s for %s/%s repeats a note as a paragraph:\n%s", name, dispatch.Item, dispatch.Step, text)
		}
	}
}
